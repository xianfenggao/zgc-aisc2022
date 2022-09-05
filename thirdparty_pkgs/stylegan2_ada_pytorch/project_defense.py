import dnnlib
import torch
import numpy as np
import logging
import sys

sys.path.append('../../')

from StyleGAN2 import StyleGAN2
from torch.nn import functional as F


def project(
        targets: torch.Tensor,  # [N,C,H,W] and dynamic range [0,255], W & H must match G output resolution,
        num_steps=5000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        l1_weight=1e-3,
        device='cuda'
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

    stylegan = StyleGAN2()
    noise_bufs = stylegan.noise_bufs
    w_avg = stylegan.w_avg
    w_std = stylegan.w_std
    G = stylegan.G

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    with open('./vgg16.pt', 'rb') as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = targets.to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(np.repeat(np.repeat(w_avg, (len(target_images)), 0), 18, 1), dtype=torch.float32,
                         device=device, requires_grad=True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
        synth_images = torch.clamp(synth_images, 0., 255)

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        # L1 loss
        l1_dist = torch.norm((synth_images - target_images) / 255, 1)

        loss = dist + l1_weight * l1_dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            logging.info(
                f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f}, l1_dist {l1_dist}, loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out, synth_images.detach().cpu()


if __name__ == '__main__':
    from torchvision import transforms as trans
    from PIL import Image
    import os

    totensor = trans.Compose([
        trans.Resize((1024, 1024)),
        trans.ToTensor(),
        trans.Lambda(lambda x: x * 255)
    ])
    # os.system(f'cp /apdcephfs/private_xianfenggao/Code/FEP/output/eval_transferability/eval_api_demo2/single/ArcFace_torch_r100_glint360k/0001_adv.jpg ./ori.png')
    im = totensor(Image.open('/apdcephfs/private_xianfenggao/Code/FEP/Tdata/zhangyao2/aligned/16184774844791.png'))
    _, imgs = project(targets=im.unsqueeze(0))
    Image.fromarray(imgs[0].numpy().transpose([1, 2, 0]).astype('uint8')).save('projected.png')
