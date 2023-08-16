import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
import scipy.io as sio
from pdb import set_trace as stx
import math
from skimage.metrics import structural_similarity as compare_ssim

##########################################
# 01. LOAD YAML
##########################################
yaml_file = 'Options/RealDenoising_Restormer_small.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r', encoding='UTF8'), Loader=Loader)

s = x['network_g'].pop('type')


##########################################
# 02. LOAD MODEL AND WEIGHT
##########################################
model_restoration = Restormer(**x['network_g'])

# Origin Pretrianed Weight
# checkpoint = torch.load('./pretrained_models/real_denoising.pth')
# model_restoration.load_state_dict(checkpoint['params'])

# Fine-Tuned Weight
model_restoration.load_state_dict(torch.load('./pts/0_9fine_tuned_model12.pt'))

# GPU
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

##########################################
# 03. EVALUATE
'''
    CHEAST1, [8,10]을 기준으로 진행
'''
##########################################

origin = cv2.cvtColor(cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test_patches_2/origin/Chest1(3072_3072)_IP.dcm.png/8_10.png'), cv2.COLOR_BGR2RGB)
noisy = cv2.cvtColor(cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test_patches_2/gaussian/Chest1(3072_3072)_IP.dcm.png/8_10.png'), cv2.COLOR_BGR2RGB)

cv2.imwrite('./test_if_training_noisy.png', noisy)
cv2.imwrite('./test_if_training_origin.png', origin)

pretrained_denoising = cv2.cvtColor(cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test_patches_2/gaussian/Chest1(3072_3072)_IP.dcm.png/8_10.png'), cv2.COLOR_BGR2RGB)

input_ = torch.from_numpy(noisy).float().div(255.).permute(2,0,1).unsqueeze(0)
input_.cuda()
print(input_.shape)
    
with torch.no_grad():
    restored = model_restoration(input_) # input (1, 1, 256, 256)

    restored = restored.cuda()
    res_img = restored.transpose(0, 2).transpose(1, 3).transpose(2, 3).squeeze()
    print(res_img.shape)
    res_img = res_img.cpu()
    res_image = (res_img).detach().numpy()
    pred_img = np.uint8(res_image * 255.0)
    
    # IF COLOR
    cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)

    cv2.imwrite('./test_if_training_origin_result_model7_origin.png', pred_img)
    
    
    
# IF COLOR
cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)








def psnr_between(origin, denoised):
    mse = np.mean( (origin - denoised) ** 2 ) #MSE
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) #PSNR


def ssim_between(origin, denoised):

    score, diff = compare_ssim(origin, denoised, full=True)
    diff = (diff * 255).astype("uint8")
    return score

# 영상 하나 test하고 visualize하기

noisy = cv2.imread('./test_if_training_noisy.png', cv2.IMREAD_GRAYSCALE)
print(noisy)
origin = cv2.imread('./test_if_training_origin.png', cv2.IMREAD_GRAYSCALE)
denoised = cv2.imread('./test_if_training_origin_result_model7_origin.png', cv2.IMREAD_GRAYSCALE)

print("[BEFORE]")
print("psnr : ", psnr_between(origin, noisy))
print("ssim : ", ssim_between(origin, noisy))

print("[AFTER]")
print("psnr : ", psnr_between(origin, denoised))
print("ssim : ", ssim_between(origin, denoised))

print("[SAME]")
print("psnr : ", psnr_between(denoised, denoised))
print("ssim : ", ssim_between(denoised, denoised))

cv2.imshow('origin', origin)
cv2.imshow('denoised', denoised)
cv2.imshow('noisy', noisy)

cv2.waitKey(0)
#cv2.imwrite('denoised_8.png', denoised)