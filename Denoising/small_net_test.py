# import packages
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 최신 CPU라 더 빨라질수 있다는데.. 어차피 CPU연산 안함
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2

import torch
from basicsr.models.archs.restormer_arch import Restormer
from pdb import set_trace as stx
import math
from skimage.metrics import structural_similarity as compare_ssim

from torch.utils.data import DataLoader
import gc
import tensorflow as tf
from util import *

# GPU 할당량 최대치의 60%로 제한 => 발열 제어
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config=config)

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
model_restoration.train()

# first step
#checkpoint = torch.load('./pretrained_models/real_denoising.pth')
#model_restoration.load_state_dict(checkpoint['params'], strict=False)
    # strict=False로 인해 가능한 weight만 불러오게됨

# second step
model_restoration.load_state_dict(torch.load('./pts/0_0fine_tuned_model11.pt'), strict=False)

model_restoration.cuda()
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.eval()

# model.named_parameters() 로 name을 확인하고 아래와 같이 required_grad = True 로 변경
# for name, param in model_restoration.named_parameters():
#    if 'attn' in name :
#        param.requires_grad = True
#        print(name)
        


##########################################
# 03. EVALUATE
'''
    CHEAST1, [8,10]을 기준으로 진행
'''
##########################################

origin = cv2.cvtColor(cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test_patches/origin/Chest1(3072_3072)_IP.dcm.png/8_10.png'), cv2.COLOR_BGR2RGB)
noisy = cv2.cvtColor(cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test_patches/gaussian/Chest1(3072_3072)_IP.dcm.png/8_10.png'), cv2.COLOR_BGR2RGB)

cv2.imwrite('./test_if_training_noisy.png', noisy)
cv2.imwrite('./test_if_training_origin.png', origin)

pretrained_denoising = cv2.cvtColor(cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test_patches/gaussian/Chest1(3072_3072)_IP.dcm.png/8_10.png'), cv2.COLOR_BGR2GRAY)

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