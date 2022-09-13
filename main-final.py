from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from locale import normalize
import os
import random
import sys
import time
import warnings

import cv2
import matplotlib
import numpy as np
from sympy import im
import torch
import torchvision
from collections import namedtuple
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as trans
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

sys.path.append('./')

from models.FaceModels import getmodels
from models.FaceDetector import RetinaFace_ResNet50
from thirdparty_pkgs.stylegan2_ada_pytorch.StyleGAN2 import StyleGAN2
from utils.misc import clear_requires_grad, read_image, Logger, create_mask, TVLoss, manual_all_seeds, batch_forward
from utils_ import islegal_mask
from utils.data_augmentation.transformations import Translate, Rotate, Hflip, Scale, GaussianBlur2d

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

# data_path parameters
parser.add_argument('-sp', '--source_path', type=str, default='dataset/Face/zgc-aisc2022/data')
parser.add_argument('-tp', '--target_path', type=str, default='dataset/Face/zgc-aisc2022/data')
parser.add_argument('--output_dir', type=str, default='./output/zgc_aisc/demo_stylegan', help='Output directory.')

parser.add_argument('--mask_path', default='dataset/Face/zgc-aisc2022/mask/test.png', help='')
parser.add_argument('--image_shape', default=112, type=int, )

parser.add_argument('--src_models', type=str, default='TFace_IR101_Aug_glint360k,MXNET_LResNet50E_IR_SGM,MXNET_LResNet100E_IR_SGM', help='White-box model')
parser.add_argument('--eval_models', type=str, default='TFace_IR_SE_50_2data,ArcFace_torch_ir50_glint360k,'\
                    'TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_IR101_Aug_glint360k,'\
                    'TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,' \
                    'FaceNet_vggface2,evoLVe_IR_50_Asia,TFace_DenseNet201_splitbn_faceemore_glintasia')
parser.add_argument('--train_with_eval', action='store_true', default=True)
parser.add_argument('--eval_iter', type=int, default=10)
parser.add_argument('--loss_type', default='l2', choices=['l2', 'cos'])

parser.add_argument('--num_iter', type=int, default=100, help='Number of iterations.')
parser.add_argument('--batch_size', type=int, default=5, help='')
parser.add_argument('--adv_nums', type=int, default=5, help='')

# input diversity parameters
parser.add_argument('--rotate', type=float, default=0., help='rotate?')
parser.add_argument('--translate', type=float, default=0., help='project?')
parser.add_argument('--diversity_shift', type=int, default=0, help='mask diversity amount')
parser.add_argument('--latent_noise', default=0, type=float, help='std of noise added to latent each epoch')

parser.add_argument('--attack_loss', help='data set attack loss function', type=str, default='feature_loss')
parser.add_argument('--feat_weight', help='feat_weight', type=float, default=188)
parser.add_argument('--layer_num', type=int, default=0, help='nps')
parser.add_argument('--verbose', default=1, type=int, choices=[0, 1])

parser.add_argument('--use_tv', default=0, type=int, help='')
parser.add_argument('--tv_weight', default=1e-3, type=float, help='')

parser.add_argument('--color_transform', type=int, default=0, help='whether use')
parser.add_argument('--reg_loss_weight', default=0, type=float)

parser.add_argument('--opt', default='adam', help='adam or sgd?')
parser.add_argument('--lr', default=1e-2, type=float, help='')
parser.add_argument('--eval_local', default=0, type=int, help='')
parser.add_argument('--abort_early', default=1, type=int, help='0 for False else True')
parser.add_argument('--weight_decay', default=0, type=float,)
parser.add_argument('--truncation_psi', default=0.66, type=float)
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

log_file = Logger(os.path.join(args.output_dir, 'log.log'))
feat_list = []
register_handle = []
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

log_file.print(str(args))

seed = 11111
manual_all_seeds(seed)


def feature_hook(module, input, output):
    global feat_list
    norm = torch.sqrt(torch.sum(output ** 2, dim=(1, 2, 3), keepdim=True))
    norm_output = output / norm
    feat_list.append(norm_output)


def main():
    output_dir = args.output_dir
    loss_type=args.loss_type
    global feat_list
    global register_handle

    # load models 
    face_models = args.src_models.split(',')
    models = getmodels(face_models)
    clear_requires_grad(models)
    
    if args.train_with_eval:
        eval_models = getmodels(args.eval_models.split(','))
        clear_requires_grad(eval_models)
        # args.eval_iter = 1

    # register intermideate feature
    for model in models:
        net = model.backbone
        # if len(register_handle) == 0:
        feat_names = []
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.stride == (2, 2):
                register_handle.append(module.register_forward_hook(feature_hook))
    
    # create masks for attack images
    mask = create_mask(args.mask_path, image_shape=(args.image_shape,args.image_shape))
    assert islegal_mask(mask[:, :, 0]), mask.shape
    projection_mask = torch.Tensor(mask.transpose(2, 0, 1)[None, :]).to(device).repeat(args.batch_size, 1, 1, 1)

    # load stylegan2 model
    stylegan2_torch = StyleGAN2('./ckpts/FaceModels/GAN/stylegan/ffhq.pkl',
                                regularize_noise_weight=args.reg_loss_weight).eval().cuda()
    th_latent = stylegan2_torch.w_avg.clone().detach().repeat([args.batch_size, 1, 1]).to(device)
    th_latent.requires_grad = True

    # define optimizer
    var_list = [th_latent] + list(stylegan2_torch.noise_bufs.values())

    with torch.no_grad():
        ori_img = stylegan2_torch.forward(th_latent.data)
        ori_img = F.interpolate(ori_img, (256, 256), mode='bilinear')
        canvas = ori_img
        detector = RetinaFace_ResNet50()
        align_grid = detector.get_batch_grids(ori_img, 256)

    tv = TVLoss()
    cur_output_dir = os.path.join(args.output_dir, 'images')
    if not os.path.exists(cur_output_dir):
        os.makedirs(cur_output_dir)
    
    import lpips, timm
    lpipses = [
        lpips.LPIPS(net='squeeze').to(device),
        lpips.LPIPS(net='vgg').to(device),
        # lpips.LPIPS(net='alex').to(device),
            
    ]
    Lpips = lambda x, y : sum([l.forward(x, y, normalize=True) for l in lpipses]) / len(lpipses)
    for model in lpipses:
        net = model.net
        # if len(register_handle) == 0:
        feat_names = []
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.stride == (2, 2):
                register_handle.append(module.register_forward_hook(feature_hook))

    
    # input_func = lambda x: Rotate(Translate(Hflip(x, 0.2), args.translate), args.rotate)
    # input_func = lambda x: x + (torch.rand_like(x) - 0.5) * 2 * 10 / 255
    input_func = lambda x: x 
    # input_func = lambda x: Scale((Hflip(x, 0.5)), 0.2)
    attacker_images = []
    for path in sorted(os.listdir(args.source_path)):
        img_path = os.path.join(args.source_path, path)
        attacker_images.append(read_image(img_path, (args.image_shape, args.image_shape)) / 255.)
    source_imgs = torch.from_numpy(np.array(attacker_images).transpose([0, 2, 3, 1]))
    
    attacker_images = []
    for path in sorted(os.listdir(args.target_path)):
        img_path = os.path.join(args.target_path, path)
        attacker_images.append(read_image(img_path, (args.image_shape, args.image_shape)) / 255.)
    target_imgs = torch.from_numpy(np.array(attacker_images).transpose([0, 2, 3, 1]))

    source_imgs = source_imgs.cuda()
    target_imgs = target_imgs.cuda()

    # extract victim feature and take them as loss function target
    target_features = list()
    if args.attack_loss is not None and 'feature_loss' in args.attack_loss:
        inter_attack_feats = list()
        feat_list = list()

    for model in models:
        target_features.append(model(target_imgs))
        if args.attack_loss is not None and 'feature_loss' in args.attack_loss:
            inter_attack_feats = inter_attack_feats + [in_feat.mean(0, keepdim=True) for in_feat in feat_list]
            feat_list = list()

    th_latent.data = stylegan2_torch.w_avg.repeat([args.batch_size, 1, 1]).to(device)
    stylegan2_torch.init_noises()

    # original_vars_data = [var.clone().detach() for var in var_list]
    if args.opt == 'adam':
        opt = torch.optim.Adam(params=var_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        opt = torch.optim.SGD(params=var_list, lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        opt = torch.optim.Adam(params=var_list, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    # attack
    last_total_loss = 1e9  # used to abort early
    losses = []
    xs = []
    eval_score_x = []
    eval_score_y = []
    for k in range(1, 1+args.num_iter):

        cur_iter_total_loss = 0
        # input_diversity params set for ensemble
        scale_list = (0, 2, 1., -1, -2)
        
        loss_cur = 0
        opt.zero_grad()

        syn_images = stylegan2_torch.forward(th_latent, args.truncation_psi)

        # align
        canvas = F.interpolate(syn_images, size=(256, 256), mode='bilinear')
        canvas = F.grid_sample(canvas, align_grid.data, mode='bilinear')
        canvas = F.interpolate(canvas, size=(args.image_shape, args.image_shape), mode='bilinear')

        image_aligned = torch.clamp(source_imgs.data * (1 - projection_mask) + canvas * projection_mask, min=0, max=1.)
        image_adv = input_func(image_aligned)
        
        # loss 1
        if args.use_tv:
            loss1 = tv(image_adv) * args.tv_weight
        else:
            loss1 = 0

        if args.color_transform:
            # add color robustness trick
            # randomly choose an channel to add uniform noise
            rn_channel = torch.randint(0, 3, size=())
            color_op = torch.zeros_like(image_adv)
            color_op[:, rn_channel, :, :] = torch.rand(*image_adv.shape)[:, rn_channel, :, :] * 10 - 5
            color_op[color_op < 0] = color_op[color_op < 0] - 15
            color_op[color_op > 0] = color_op[color_op > 0] + 15
            image_adv = image_adv + color_op

        # forward to get features and embeddings
        feats_adv = list()
        if args.attack_loss is not None and 'feature_loss' in args.attack_loss: inter_src_feats = list()
        for idx, model in enumerate(models):
            model.zero_grad()
            if args.attack_loss is not None and 'feature_loss' in args.attack_loss: feat_list = list()
            feat_adv = model(image_adv)
            if k % args.eval_iter == 0:
                with torch.no_grad():
                    if not args.train_with_eval:
                        cos_simi = torch.sum(model(image_aligned) * target_features[idx], dim=1).mean().item()
                        log_file.print('iter\t' + str(k) + '\t' + model.name + '\tcos similarity\t' + str(cos_simi))
                    elif idx == 0:
                        eval_score_x.append(k)
                        eval_score_y.append(0)
                        for eval_model in eval_models:
                            cos_simi = torch.sum(eval_model(image_aligned) * eval_model(target_imgs), dim=1).mean().item()
                            log_file.print('iter\t' + str(k)+ '\t' + eval_model.name + '\tcos similarity\t' + str(cos_simi))
                            eval_score_y[-1] += cos_simi
                        eval_score_y[-1] /= len(eval_models)

            feats_adv.append(feat_adv)
            if args.attack_loss is not None and 'feature_loss' in args.attack_loss:
                inter_src_feats = inter_src_feats + feat_list

        # loss2
        loss2 = 0
        for src_feat, dst_feat in zip(feats_adv, target_features):
            if loss_type == 'l2':
                loss2 += torch.mean(torch.sum((src_feat - dst_feat.data) ** 2, dim=(-1)))
            elif loss_type == 'cos':
                loss2 += -torch.sum(src_feat * dst_feat.data)  # Cos dist
            # mean_src_feat = torch.mean(src_feat.detach().data, dim=0).expand_as(src_feat)
            # loss += 0.1 * torch.mean(torch.sum((src_feat - mean_src_feat.data) ** 2), dim = (-1))
        loss2 = loss2 / len(target_features) * 3

        # loss3
        loss3 = 0
        if args.attack_loss is not None and 'feature_loss' in args.attack_loss:
            for inter_src_feat, inter_dst_feat in zip(inter_src_feats, inter_attack_feats):
                loss3 = loss3 + torch.mean(
                    torch.sum((inter_src_feat - inter_dst_feat.data) ** 2, dim=(-1)))
            loss3 = loss3 / len(inter_dst_feat) * args.feat_weight
        
        # noise regularization loss
        loss4 = stylegan2_torch.regularization_noises()
        
        # loss7 lpips
        loss1 = Lpips(target_imgs.data, image_adv).mean()

        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        loss_cur += loss.item()
        # for i in range(len(var_list)):
        #     grads_sum[i] += var_list[i].grad.data
        cur_iter_total_loss += loss.item()

        # update grad
        # for i in range(len(var_list)):
        #     grads_sum[i] += var_list[i].grad.data

        opt.step()

        # print('latent distance:', [torch.norm(original_var.data - var.data, 2).item() for original_var, var in zip(original_vars_data, var_list)][:4])
        # if args.verbose:
        #     print("Iter %3d\t loss1 %.4f\tloss2 %.4f\tloss3 %.4f\tloss4 %.4f\tloss %.4f" % (
        #         k, loss1, loss2, loss3, loss4, loss))
        log_file.print("Iter %3d\t loss1 %.4f\tloss2 %.4f\tloss3 %.4f\tloss4 %.4f\tloss %.4f" % (
            k, loss1, loss2, loss3, loss4, loss))
        loss_cur *= len(models) * 10
        xs.append(k)
        losses.append(loss_cur)
        plt.figure()
        # plt.ylim(0,1) #同上
        plt.plot(xs, losses)
        if args.train_with_eval:
            plt.plot(eval_score_x, eval_score_y)
            plt.legend(['loss', 'eval_score'])
            # print(eval_score_y)
        plt.grid()
        plt.savefig(os.path.join(output_dir, 'loss.png'))
        plt.close()

        if args.abort_early and k > 200 and k % args.eval_iter == 0:
            # abort early
            if args.train_with_eval and eval_score_y[-1]*0.99 <= np.mean(eval_score_y[-5:]) and losses[-1] >= np.mean(losses[-5:])*0.99:
                log_file.print(f'early aborted after {k} iterations')
                break

    image_aligned = (255*image_aligned).detach().cpu().numpy().transpose([0, 2, 3, 1]).astype('uint8')
    canvas[projection_mask == 0] = 1.  # for white background
    canvas = canvas*255
    canvas = canvas.clone().detach().cpu().numpy().transpose([0, 2, 3, 1]).astype('uint8')
    Image.fromarray(canvas[0]).save(os.path.join(cur_output_dir, 'canvas.png'))


if __name__ == '__main__':
    main()
    



