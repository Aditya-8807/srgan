import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

def calculate_psnr(img1, img2, data_range=1.0):
    if isinstance(img1, torch.Tensor): img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor): img2 = img2.detach().cpu().numpy()
    if img1.ndim == 4:
        psnrs = [psnr_metric(img1[b].transpose(1, 2, 0), img2[b].transpose(1, 2, 0), data_range=data_range) for b in range(img1.shape[0])]
        return np.mean(psnrs)
    return psnr_metric(img1.transpose(1, 2, 0), img2.transpose(1, 2, 0), data_range=data_range)

def calculate_ssim(img1, img2, data_range=1.0):
    if isinstance(img1, torch.Tensor): img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor): img2 = img2.detach().cpu().numpy()
    if img1.ndim == 4:
        ssims = [ssim_metric(img1[b].transpose(1, 2, 0), img2[b].transpose(1, 2, 0), data_range=data_range, channel_axis=-1) for b in range(img1.shape[0])]
        return np.mean(ssims)
    return ssim_metric(img1.transpose(1, 2, 0), img2.transpose(1, 2, 0), data_range=data_range, channel_axis=-1)
