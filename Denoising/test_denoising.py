import numpy as np
import cv2
import torch
import torch.nn as nn
import argparse

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from basicsr.models.archs.restormer_arch import Restormer
from utils import divide_patches, concat_h, concat_v, psnr_between, ssim_between


##########################################
# 01. PREPARE DATA
##########################################
parser = argparse.ArgumentParser(description='Grayscale Denoising using fine-tuned Restormer')

parser.add_argument('--origin_path', default='C:/Users/NDT/Desktop/Image_denoising/Data/test/origin/Chest4(2957_2887)_IP.dcm.png', type=str, help='Path of noisy images')
parser.add_argument('--noisy_path', default='C:/Users/NDT/Desktop/Image_denoising/Data/test/gaussian/Chest4(2957_2887)_IP.dcm.png', type=str, help='Path of noisy images')
parser.add_argument('--result_path', default='./', type=str, help='Path for save denoised image')
parser.add_argument('--pretrained', default=False, type=bool, help='If you want to test pretrained, not fine tuned weight')
args = parser.parse_args()


origin = cv2.imread(args.origin_path, cv2.COLOR_BGR2GRAY)
noisy = cv2.imread(args.noisy_path, cv2.COLOR_BGR2GRAY)

noisy_patches = divide_patches(noisy, patch_size=512, overlap_size=85)
denoised_patches = np.zeros_like(noisy_patches)


##########################################
# 02. DENOISING
##########################################
# 1) Import Yaml
yaml_file = 'RealDenoising_Restormer_small.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r', encoding='UTF8'), Loader=Loader)
s = x['network_g'].pop('type')

# 2) Load Model and Weight
model_restoration = Restormer(**x['network_g'])

if args.pretrained:
    model_restoration.load_state_dict(torch.load('Pretrained.pt')['params'], strict=False)
else:
    model_restoration.load_state_dict(torch.load('Finetuned.pt')) 

model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 3) Evaluate
input_ = torch.from_numpy(noisy_patches).float().div(255.).cuda()

for i in range(input_.shape[0]):
    for j in range(input_.shape[1]):
        
        with torch.no_grad(): 
            denoised_patch = model_restoration(input_[i][j].unsqueeze(0).unsqueeze(0)) # input(1, 3, 256, 256), output(1, 3, 256, 256)
        denoised_patch = denoised_patch * 255.0
        denoised_patch = torch.clamp(denoised_patch, 0, 255).permute(0, 2, 3, 1).squeeze(0).cpu().detach().numpy()
        denoised_patches[i][j] = denoised_patch.squeeze(2)
        
        #cv2.imshow('00', denoised_patches[i][j])
        #cv2.waitKey(0)
        
denoised_patches = np.uint8(denoised_patches)


# 4) To one Image
horizontal_concatenated = []
for i in range(denoised_patches.shape[0]):
    row = denoised_patches[i][0]
    for j in range(1, denoised_patches.shape[1]):
        row = concat_h(row, denoised_patches[i][j])
    horizontal_concatenated.append(row)

final_image = horizontal_concatenated[0]
for i in range(1, len(horizontal_concatenated)):
    final_image = concat_v(final_image, horizontal_concatenated[i])
    
denoised = final_image[:origin.shape[0], :origin.shape[1]]

cv2.imwrite(os.path.join(args.result_path, args.noisy_path+'__DENOISED.png'), denoised)


##########################################
# 03. PRINT RESULT
##########################################
print("[BEFORE]")
print("psnr : ", psnr_between(origin, noisy))
print("ssim : ", ssim_between(origin, noisy))

print("[AFTER]")
print("psnr : ", psnr_between(origin, denoised))
print("ssim : ", ssim_between(origin, denoised))