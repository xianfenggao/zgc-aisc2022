import argparse
import logging
# import timm
import os
import sys

import torch
import torch.cuda.amp as amp
import torch.distributed as dist

sys.path.append('./')
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

from Tdata.datasets import IterationDataloader
from Tdata.mxfacedataset import MXFaceDataset
from util.utils import AverageMeter, Timer
from util.utils import adjust_learning_rate, warm_up_lr
from util.utils import accuracy_dist
from util.distributed_functions import AllGather
from loss import get_loss
from task.base_task import BaseTask
from util.split_batchnorm import convert_splitbn_model
from augmentation.transforms import Rotate, ShearXY, AutoContrast, Brightness, Sharpness, TranslateXY, Cutout

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


# os.environ['WORLD_SIZE'] = '1'
# os.environ['RANK'] = '0'
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '35467'

class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """

    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)
        self.elastic_resume = False

    def _loop_step(self, train_loaders, backbone, heads, criterion, opt,
                   scaler, epoch, class_splits):
        """ load_data --> extract feature --> calculate loss and apply grad --> summary
        """
        backbone.train()  # set to training mode
        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes

        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for batch, samples in enumerate(zip(*train_loaders)):
            global_batch = epoch * self.step_per_epoch + batch
            if global_batch <= self.warmup_step:
                warm_up_lr(global_batch, self.warmup_step, self.cfg['LR'], opt)
            if batch >= self.step_per_epoch:
                break

            inputs = torch.cat([x[0] for x in samples], dim=0)
            inputs = inputs.cuda(non_blocking=True)
            labels = torch.cat([x[1] for x in samples], dim=0)
            labels = labels.cuda(non_blocking=True)

            if self.cfg['AMP']:
                with amp.autocast():
                    features = backbone(inputs)
                features = features.float()
            else:
                features = backbone(inputs)

            # gather features
            _features_gather = [torch.zeros_like(features) for _ in range(self.world_size)]
            features_gather = AllGather(features, *_features_gather)
            features_gather = [torch.split(x, batch_sizes) for x in features_gather]
            all_features = []
            for i in range(len(batch_sizes)):
                all_features.append(torch.cat([x[i] for x in features_gather], dim=0).cuda())

            # gather labels
            labels_gather = [torch.zeros_like(labels) for _ in range(self.world_size)]
            dist.all_gather(labels_gather, labels)
            labels_gather = [torch.split(x, batch_sizes) for x in labels_gather]
            all_labels = []
            for i in range(len(batch_sizes)):
                all_labels.append(torch.cat([x[i] for x in labels_gather], dim=0).cuda())

            step_losses = []
            step_original_outputs = []
            for i in range(len(batch_sizes)):
                outputs, part_labels, original_outputs = heads[i](all_features[i], all_labels[i])
                step_original_outputs.append(original_outputs)
                loss = criterion(outputs, part_labels) * self.branch_weights[i]
                step_losses.append(loss)

            total_loss = sum(step_losses)
            # compute gradient and do SGD step
            opt.zero_grad()
            # Automatic Mixed Precision setting
            if self.cfg['AMP']:
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                total_loss.backward()
                opt.step()

            for i in range(len(batch_sizes)):
                # measure accuracy and record loss
                prec1, prec5 = accuracy_dist(self.cfg,
                                             step_original_outputs[i].data,
                                             all_labels[i],
                                             class_splits[i],
                                             topk=(1, 5))

                am_losses[i].update(step_losses[i].data.item(),
                                    all_features[i].size(0))
                am_top1s[i].update(prec1.data.item(), all_features[i].size(0))
                am_top5s[i].update(prec5.data.item(), all_features[i].size(0))
                # wirte loss and acc to tensorboard
                summarys = {
                    'train/loss_%d' % i: am_losses[i].val,
                    'train/top1_%d' % i: am_top1s[i].val,
                    'train/top5_%d' % i: am_top5s[i].val
                }
                self._writer_summarys(summarys, batch, epoch)

            duration = t.get_duration()
            self._log_tensor(batch, epoch, duration, am_losses, am_top1s, am_top5s)

    def _prepare(self):
        # train_loaders, class_nums = self._make_inputs()
        class_nums = [self.cfg['DATASETS'][i]['class_nums'] for i in range(len(self.cfg['DATASETS']))]

        train_loaders = []

        augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            Rotate(.5, .05),
            ShearXY(0.5, 0.1),
            TranslateXY(0.5, .1),
            Cutout(.3, .8),
            AutoContrast(.5, .1),
            Sharpness(.3, .1),
            Brightness(.5, .12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        for i in range(len(self.cfg['DATASETS'])):

            dataset = MXFaceDataset(self.cfg['DATASETS'][i]['DATA_ROOT'], self.rank,
                                    trans=augment)
            batch_size = self.batch_sizes[i]
            if i > 0:
                # ensure all batch_size is the same so that we can split them the same into different bn layers easily
                assert batch_size == self.batch_sizes[i - 1]
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True)
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(sampler is None),
                num_workers=self.cfg['NUM_WORKERS'],
                pin_memory=True,
                sampler=sampler,
                drop_last=True)

            train_loaders.append(train_loader)
            # print(self.cfg['DATASETS'][i]['DATA_ROOT'], int(len(dataset) / (batch_size * self.world_size)))
            self.step_per_epoch = max(
                self.step_per_epoch,
                int(len(dataset) / (batch_size * self.world_size)))
        train_loaders = [
            IterationDataloader(train_loader, self.step_per_epoch * self.epoch_num, 0)
            for train_loader in train_loaders]

        backbone, heads, class_splits = self._make_model(class_nums)

        # convert to splitbn #
        if len(train_loaders) > 1 and self.cfg['SPLITBN']:
            backbone = convert_splitbn_model(backbone, len(train_loaders))

        if not self.elastic_resume:
            self._load_pretrain_model(backbone, self.cfg['BACKBONE_RESUME'], heads, self.cfg['HEAD_RESUME'])
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[self.local_rank])
        loss = get_loss('DistCrossEntropy').cuda()
        opt = self._get_optimizer(backbone, heads)
        scaler = amp.GradScaler()
        self._load_meta(opt, scaler, self.cfg['META_RESUME'])
        return train_loaders, backbone, heads, class_splits, loss, opt, scaler

    def train(self):
        """ make_inputs --> make_model --> load_pretrain --> build DDP -->
            build optimizer --> loop step
        """
        train_loaders, backbone, heads, class_splits, loss, opt, scaler = self._prepare()
        self._create_writer()
        for epoch in range(self.start_epoch, self.epoch_num):
            adjust_learning_rate(opt, epoch, self.cfg)
            self._loop_step(train_loaders, backbone, heads, loss, opt, scaler, epoch, class_splits)
            self._save_ckpt(epoch, backbone, heads, opt, scaler)

    def check_ckpt_dir(self):
        if os.path.exists('./ckpt'):
            for e in range(1, self.cfg['NUM_EPOCH']):
                if os.path.exists('ckpt/Backbone_Epoch_{}_checkpoint.pth'.format(e)) \
                        and os.path.exists('ckpt/HEAD_Epoch_{}_Split_{}_checkpoint.pth'.format(e, self.world_size - 1)):
                    continue
                else:
                    break
            self.start_epoch = e - 1
            e = e - 1
            self.cfg['BACKBONE_RESUME'] = 'ckpt/Backbone_Epoch_{}_checkpoint.pth'.format(e)
            self.cfg['HEAD_RESUME'] = 'ckpt/HEAD_Epoch_{}'.format(e)

            if e:
                self.elastic_resume = True


def main():
    parser = argparse.ArgumentParser(description='Train a TFace model')
    parser.add_argument('--config',
                        default='./thirdparty_pkgs/TFace/config/train_with_augmentation.yaml')
    parser.add_argument('--local_rank', default=0)
    args = parser.parse_args()

    task = TrainTask(args.config)
    task.init_env()
    task.check_ckpt_dir()
    task.train()


if __name__ == '__main__':
    main()
