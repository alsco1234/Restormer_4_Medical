## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09481

import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx



parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/SIDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Real_Denoising/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/real_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/RealDenoising_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)


print("start load model")
model_restoration = Restormer(**x['network_g'])

print("start load weight")
checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
#model_restoration = nn.DataParallel(model_restoration)
#model_restoration.eval()

# Process data
filepath = 'C:/Users/NDT/Desktop/Image_denoising/Restormer/Denoising/Datasets/test/SIDD/dental5(1300_1700)_IP.dcm.png'
out_dir = './'

print("start read input image")
img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
img = img[0:256, 0:256]
print(img.shape)
input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0)

restored = model_restoration(input_)
print(type(restored))
print(restored.shape)
res_img = restored.transpose(0, 2).transpose(1, 3).transpose(2, 3).squeeze()
print(res_img.shape)
res_image = (res_img).detach().numpy()
pred_img = np.uint8(res_image * 255.0)

# save denoised data
cv2.imshow("resykt", res_image)
cv2.waitKey(0)

cv2.imwrite('result.png', pred_img)


import math
from skimage.metrics import structural_similarity as compare_ssim

# 원본과 denoising간의 psnr, ssim 계산

def psnr_between(origin, denoised):
    mse = np.mean( (origin - denoised) ** 2 ) #MSE
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) #PSNR


def ssim_between(origin, denoised):
    gray_origin = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    score, diff = compare_ssim(gray_origin, gray_denoised, full=True)
    diff = (diff * 255).astype("uint8")
    return score


origin = cv2.imread('../../Data/train/origin/00000.png')
origin = origin[0:256, 0:256]

noisy = cv2.imread('../../Data/train/gaussian/00000.png')
noisy = noisy[0:256, 0:256]

denoised = cv2.imread('result.png')
denoised = denoised[0:256, 0:256]

print("[BEFORE]")
print("psnr : ", psnr_between(origin, noisy))
print("ssim : ", ssim_between(origin, noisy))

print("[AFTER]")
print("psnr : ", psnr_between(origin, denoised))
print("ssim : ", ssim_between(origin, denoised))