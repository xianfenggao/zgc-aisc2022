import argparse
import logging
# import timm
import os
import sys
import warnings

warnings.filterwarnings(action='ignore')

import torch.nn.init as init
import torch
import torch.cuda.amp as amp
import torch.distributed as dist

sys.path.append('./')
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from Tdata.datasets import IterationDataloader
from Tdata.pet_biometric_challenge_2022 import PetBiometric
from util.utils import AverageMeter, Timer
from util.utils import adjust_learning_rate, warm_up_lr
from util.utils import accuracy_dist
from util.distributed_functions import AllGather
from loss import get_loss
from task.base_task import BaseTask
from util.split_batchnorm import convert_splitbn_model
from backbone import get_model
from util.utils import get_class_split
from head import get_head

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
from sklearn.metrics import roc_auc_score
from torchsampler import ImbalancedDatasetSampler   # https://github.com/ufoym/imbalanced-dataset-sampler


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
            inputs = inputs + torch.randn_like(inputs)*20/255  # add random noise each epoch to avoid overfitting
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
                # outputs, part_labels, original_outputs = heads[i](all_features[i], all_labels[i])
                outputs, original_outputs = heads[i](all_features[i], all_labels[i])
                step_original_outputs.append(original_outputs)
                loss = criterion(outputs, all_labels[i]) * self.branch_weights[i]
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

        # self.validation_test(backbone, epoch)
        # val
        # am_top1v = [AverageMeter() for _ in batch_sizes]
        # am_top5v = [AverageMeter() for _ in batch_sizes]
        # with torch.no_grad():
        #     for im, labels in self.val_loader:
        #         im = im.cuda()
        #         labels = labels.cuda()
        #         _, pred = heads[0](backbone(im), labels)
        #         prec1, prec5 = accuracy(pred, labels, (1, 5))
        #         am_top1v[0].update(prec1.data.item(), len(im))
        #         am_top5v[0].update(prec5.data.item(), len(im))
        #     print('#################\tEpoch {}, validation top1 {}, validation top5 {}'.format(epoch, am_top1v[0].val, am_top5v[0].val))

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

    @torch.no_grad()
    def validation_test(self, backbone, epoch):
        backbone.eval()
        pairs = self.val_pairs
        num = len(pairs[-1])
        batch_num = num // 128
        result_cos = np.empty(shape=(batch_num*128))
        gt = 1 - np.array(pairs[-1][:batch_num*128])
        for i in range(batch_num):
            batch_im0 = pairs[0][i*128:(i+1)*128].cuda()
            batch_im1 = pairs[1][i * 128:(i + 1) * 128].cuda()
            feat0 = backbone(batch_im0)
            feat1 = backbone(batch_im1)
            result_cos[i*128:(i+1)*128] = torch.cosine_similarity(feat0, feat1).cpu().numpy()

        result_cos = result_cos/2+0.5
        if any(np.isnan(result_cos)):
            logging.info('Valiation contains NaN, abort')
            sys.exit(-1)

        def KFold(n=6000, n_folds=10, shuffle=False):
            folds = []
            base = list(range(n))
            for i in range(n_folds):
                test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
                train = list(set(base) - set(test))
                folds.append([train, test])
            return folds

        def eval_acc(threshold, diff, gts):
            assert len(diff) == len(gts)
            return np.sum((diff < threshold) == gts) / len(diff)

        def find_best_threshold(thresholds, predicts, gts):
            best_threshold = best_acc = 0
            for threshold in thresholds:
                accuracy = eval_acc(threshold, predicts, gts)
                if accuracy >= best_acc:
                    best_acc = accuracy
                    best_threshold = threshold
            return best_threshold

        accuracy = []
        thd = []
        folds = KFold(n=len(gt), n_folds=10, shuffle=False)
        thresholds = np.arange(0, 1.0, 0.005)
        predicts = np.array(result_cos)
        gts = np.array(gt)
        for idx, (train, test) in enumerate(folds):
            best_thresh = find_best_threshold(thresholds, predicts[train], gts[train])
            accuracy.append(eval_acc(best_thresh, predicts[test], gts[test]))
            thd.append(best_thresh)
        logging.info('epoch {}, COS ACC={:.4f} std={:.4f} thd={:.4f}, auc {:.4f}'.format(epoch, np.mean(accuracy),
                                                                                np.std(accuracy),
                                                                                np.mean(thd), roc_auc_score(1-gt, predicts)))

    def _prepare(self):
        # train_loaders, class_nums = self._make_inputs()
        class_nums = [self.cfg['DATASETS'][i]['class_nums'] for i in range(len(self.cfg['DATASETS']))]

        train_loaders = []
        augment = transforms.Compose([
            transforms.Resize(self.cfg['INPUT_SIZE']),
            transforms.RandomAffine(30, (0.1, 0.1), (0.7, 1.3)),
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # augment = create_transform(
        #     (112, 112), True,
        #     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
        #     auto_augment='rand'
        # )
        val_transform = transforms.Compose([
            # transforms.RandomResizedCrop((112, 112)),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        for i in range(len(self.cfg['DATASETS'])):
            # if i == 0:
            #     dataset = PolyUPalmprint(self.cfg['DATASETS'][i]['DATA_ROOT'], self.rank)
            # else:
            #     dataset = TongJiPalmprint(self.cfg['DATASETS'][i]['DATA_ROOT'], self.rank)
            dataset = PetBiometric(self.cfg['DATASETS'][i]['DATA_ROOT'], self.rank)
            # dataset, val = random_split(dataset, (int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)))
            # self.val_loader = DataLoader(val, 128, num_workers=4)
            dataset.transform = augment
            # self.val_loader.transform = val_transform

            batch_size = self.batch_sizes[i]
            if i > 0:
                # ensure all batch_size is the same so that we can split them the same into different bn layers easily
                assert batch_size == self.batch_sizes[i - 1]
            # sampler = torch.utils.data.distributed.DistributedSampler(
            #     dataset,
            #     num_replicas=self.world_size,
            #     rank=self.rank,
            #     shuffle=True)
            sampler = ImbalancedDatasetSampler(dataset)
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

            # with open(os.path.join(self.cfg['DATASETS'][i]['DATA_ROOT'], 'val_pairs.txt')) as f:
            #     lines = f.readlines()
            # self.val_pairs = [[], [], []]
            # for line in lines:
            #     line = line.strip('\n').split('\t')
            #     self.val_pairs[2].append(int(line[-1]))
            #     self.val_pairs[0].append(val_transform(Image.open(os.path.join(
            #         self.cfg['DATASETS'][i]['DATA_ROOT'], 'train/images', line[0]
            #     ))))
            #     self.val_pairs[1].append(val_transform(Image.open(os.path.join(
            #         self.cfg['DATASETS'][i]['DATA_ROOT'], 'train/images', line[1]
            #     ))))
            # self.val_pairs[0] = torch.stack(self.val_pairs[0])
            # self.val_pairs[1] = torch.stack(self.val_pairs[1])


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
        loss = get_loss(self.cfg['LOSS_NAME']).cuda()
        opt = self._get_optimizer(backbone, heads)
        scaler = amp.GradScaler()
        self._load_meta(opt, scaler, self.cfg['META_RESUME'])
        return train_loaders, backbone, heads, class_splits, loss, opt, scaler

    def _make_model(self, class_nums):
        """ build training backbone and heads
        """
        backbone_name = self.cfg['BACKBONE_NAME']
        backbone_model = get_model(backbone_name)
        backbone = backbone_model(self.input_size)

        logging.info("{} Backbone Generated".format(backbone_name))

        embedding_size = self.cfg['EMBEDDING_SIZE']
        heads = []
        class_splits = []
        metric = get_head(self.cfg['HEAD_NAME'], dist_fc=self.cfg['DIST_FC'])

        for class_num in class_nums:
            class_split = get_class_split(class_num, self.world_size)
            class_splits.append(class_split)
            logging.info('Split FC: {}'.format(class_split))
            init_value = torch.FloatTensor(embedding_size, class_num)
            init.normal_(init_value, std=0.01)
            head = metric(in_features=embedding_size,
                          out_features=class_num,  # new added
                          # gpu_index=self.rank,
                          # weight_init=init_value,
                          # class_split=class_split
                          )
            del init_value
            heads.append(head)
        backbone.cuda()
        for head in heads:
            head.cuda()
        return backbone, heads, class_splits


def main():
    parser = argparse.ArgumentParser(description='Train a TFace model')
    parser.add_argument('--config', default='thirdparty_pkgs/TFace/config/train_dognose_config.yaml')
    parser.add_argument('--local_rank', default=0)
    args = parser.parse_args()

    task = TrainTask(args.config)
    task.init_env()
    # task.check_ckpt_dir()
    task.train()


if __name__ == '__main__':
    main()
