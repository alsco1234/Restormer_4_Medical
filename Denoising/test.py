import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
import torch
import torch.nn as nn
import argparse
import yaml
from basicsr.models.archs.restormer_arch import Restormer
from utils import divide_patches, merge_patches, psnr_between, ssim_between



parser = argparse.ArgumentParser(description='Grayscale Denoising using fine-tuned Restormer')

parser.add_argument('--origin_path', default='./data/test/origin/Chest1.png', type=str, help='Path of noisy images')
parser.add_argument('--noisy_path', default='./data/test/noisy/Chest1.png', type=str, help='Path of noisy images')
parser.add_argument('--result_path', default='./result/', type=str, help='Path for save denoised image')
parser.add_argument('--pretrained', default=False, type=bool, help='If you want to test pretrained, not fine tuned weight')
args = parser.parse_args()


##########################################
# 01. PREPARE DATA
##########################################
origin = cv2.imread(args.origin_path, cv2.COLOR_BGR2GRAY)
noisy = cv2.imread(args.noisy_path, cv2.COLOR_BGR2GRAY)

noisy_patches = divide_patches(noisy, patch_size=512, overlap_size=85)
denoised_patches = np.zeros_like(noisy_patches)


##########################################
# 02. LOAD MODEL
##########################################
# 1) Set Yaml
yaml_file = 'options/RealDenoising_Restormer_small.yml'

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r', encoding='UTF8'), Loader=Loader)
s = x['network_g'].pop('type')


# 2) Load Model and Weight
model_restoration = Restormer(**x['network_g'])

if args.pretrained:
    model_restoration.load_state_dict(torch.load('./pts/Pretrained.pt')['params'], strict=False)
else:
    model_restoration.load_state_dict(torch.load('./pts/Finetuned.pt')) 

model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


##########################################
# 03. DENOISING
##########################################
# 1) Evaluate
input_ = torch.from_numpy(noisy_patches).float().div(255.).cuda()

for i in range(input_.shape[0]):
    for j in range(input_.shape[1]):
        
        with torch.no_grad(): 
            denoised_patch = model_restoration(input_[i][j].unsqueeze(0).unsqueeze(0)) # input(1, 3, 256, 256), output(1, 3, 256, 256)
        denoised_patch = denoised_patch * 255.0
        denoised_patch = torch.clamp(denoised_patch, 0, 255).permute(0, 2, 3, 1).squeeze(0).cpu().detach().numpy()
        denoised_patches[i][j] = denoised_patch.squeeze(2)
        
denoised_patches = np.uint8(denoised_patches)


# 2) To one Image
final_image = merge_patches(denoised_patches)
denoised = final_image[:origin.shape[0], :origin.shape[1]]


# 3) Save Result Image
filename = (args.noisy_path.split('/')[-1]).split('.')[0]
cv2.imwrite((args.result_path + filename + '_DENOISED.png'), denoised)
print((args.result_path + filename + '_DENOISED.png'))


##########################################
# 03. PRINT RESULT
##########################################
print("[BEFORE]")
print("psnr : ", psnr_between(origin, noisy))
print("ssim : ", ssim_between(origin, noisy))

print("[AFTER]")
print("psnr : ", psnr_between(origin, denoised))
print("ssim : ", ssim_between(origin, denoised))