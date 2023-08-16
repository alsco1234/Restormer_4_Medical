"""

    Restormer / Denoising / test_one.py
    
    TODO:
    1) real-color로 all image 성능 테스트
    2) fine-tuning 구현하고 local에서 실험하기
    3) server에서 fine-tuning 실험하기
    
"""

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

import time

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
#checkpoint = torch.load('./pretrained_models/real_denoising.pth')
#model_restoration.load_state_dict(checkpoint['params'])

# Fine-Tuned Weight
model_restoration.load_state_dict(torch.load('./pts/2_9fine_tuned_model13.pt')) #최종 :  2_9, model13

print("model_restoration.cuda()")
model_restoration.cuda()
print("model_restoration = nn.DataParallel(model_restoration)")
model_restoration = nn.DataParallel(model_restoration)
print("model_restoration.eval()")
model_restoration.eval()

##########################################
# 03. EVALUATE
##########################################
folderpath = 'C:/Users/NDT/Desktop/Image_denoising/Data/test_patches_85/gaussian_sig25/Chest1(3072_3072)_IP.dcm.png'
out_dir = './sig25/Chest1(3072_3072)_IP.dcm.png/'

#os.mkdir('./sig25/Chest1(3072_3072)_IP.dcm.png/')

for patch in os.listdir(folderpath):    
    patchpath = os.path.join(folderpath, patch)
    
    # GPU 발열로 인한 제어
    # time.sleep(1)
    
    print('start path : ', patchpath)
    
    # IF COLOR
    #img = cv2.cvtColor(cv2.imread(patchpath), cv2.COLOR_BGR2RGB)
    #input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0)
    # IF GRAY
    img = cv2.cvtColor(cv2.imread(patchpath), cv2.COLOR_BGR2GRAY)
    input_ = torch.from_numpy(img).float().div(255.).unsqueeze(0).unsqueeze(0)
    input_ = input_.cuda()
    
    with torch.no_grad(): 
        restored = model_restoration(input_) # input (1, 1, 256, 256)

    restored = restored.cpu()
    res_img = restored.transpose(0, 2).transpose(1, 3).transpose(2, 3).squeeze()
    res_image = (res_img).detach().numpy()
    pred_img = np.uint8(res_image * 255.0)
    
    # IF GRAY
    # cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)

    cv2.imwrite(out_dir+patch, pred_img)