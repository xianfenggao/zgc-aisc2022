import torch
import copy
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import legacy

from torch import nn
from torch.nn import functional as F
from config.path_config import ckpts


class StyleGAN2(nn.Module):
    def __init__(self, pkl=os.path.join(ckpts, 'stylegan/ffhq.pkl'),
                 w_avg_samples=10000,
                 regularize_noise_weight=1e5,
                 device='cuda'
                 ):
        super(StyleGAN2, self).__init__()
        with open(pkl, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema']
        self.G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
        del G

        # Compute w stats.
        z_samples = torch.randn(w_avg_samples, self.G.z_dim).to(device)
        w_samples = self.G.mapping(z_samples, None)  # [N, L, C]
        self.w_avg = torch.mean(w_samples, dim=0, keepdim=True)
        self.w_std = (torch.sum((w_samples - self.w_avg) ** 2) / w_avg_samples) ** 0.5

        # Setup noise inputs.
        self.noise_bufs = {name: buf for (name, buf) in self.G.synthesis.named_buffers() if 'noise_const' in name}
        self.regularize_noise_weight = regularize_noise_weight

    def forward(self, latent, truncation_psi=0.5):
        synth_images = self.G.synthesis(latent.lerp(self.w_avg, truncation_psi), noise_mode='const')
        return torch.clamp((synth_images + 1) * .5, 0., 1.)

    def init_noises(self):
        # Init noise.
        for buf in self.noise_bufs.values():
            buf.data = torch.randn_like(buf)
            if not buf.requires_grad:
                buf.requires_grad = True

    def regularization_noises(self):
        # Noise regularization.
        reg_loss = 0.0
        for v in self.noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss * self.regularize_noise_weight


def project(target_path):
    from torchvision import transforms as trans
    from PIL import Image
    import lpips
    import logging
    import torchvision

    generator = StyleGAN2()
    generator.init_noises()
    totensor = trans.Compose([
        trans.Resize(256),
        trans.ToTensor(),
        # trans.Normalize(0.5, 0.5)
    ])

    target = totensor(Image.open(target_path)).cuda()

    Lpips = lpips.LPIPS(net='vgg').cuda()
    Lpips2 = lpips.LPIPS().cuda()
    Lpips3 = lpips.LPIPS(net='squeeze').cuda()

    opt_var = generator.w_avg.clone().detach().cuda()
    opt_var.requires_grad = True

    opt = torch.optim.Adam([opt_var] + list(generator.noise_bufs.values()), lr=0.1)

    for _ in (range(500)):
        _generated = generator.forward(opt_var)
        generated = F.interpolate(_generated, 256)
        loss1 = torch.abs(generated - target.data).mean()
        loss2 = Lpips.forward(generated, target.data, normalize=True) + \
                Lpips2.forward(generated, target.data, normalize=True) + \
                Lpips3.forward(generated, target.data, normalize=True)
        loss3 = generator.regularization_noises()
        (loss1 + loss2 + loss3).backward()
        opt.step()
        opt.zero_grad()

        logging.info('Iter %05d\tloss1 %.6f\t loss2 %.6f\t loss3 %.6f' % (_ + 1, loss1, loss2, loss3))

    torchvision.utils.save_image(_generated.detach().cpu(), 'generated.jpg')


if __name__ == '__main__':
    model = StyleGAN2()
    print(dir(model))
