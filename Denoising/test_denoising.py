import numpy as np
import cv2

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn

from basicsr.models.archs.restormer_arch import Restormer
from utils import divide_patches, concat_h, concat_v, psnr_between, ssim_between


##########################################
# 01. PREPARE DATA
##########################################
origin = cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test/origin/Chest1(3072_3072)_IP.dcm.png', cv2.COLOR_BGR2GRAY)
noisy = cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test/gaussian/Chest1(3072_3072)_IP.dcm.png', cv2.COLOR_BGR2GRAY)

noisy_patches = divide_patches(noisy, patch_size=256, overlap_size=85)
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
#model_restoration.load_state_dict(torch.load('Pre_train.pt')['params'], strict=False)
model_restoration.load_state_dict(torch.load('Finetuned.pt')) 

model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 3) Evaluate
input_ = torch.from_numpy(noisy_patches).float().div(255.)
input_ = input_.cuda()

for i in range(input_.shape[0]):
    for j in range(input_.shape[1]):
        
        with torch.no_grad(): 
            denoised_patch = model_restoration(input_[i][j].unsqueeze(0).unsqueeze(0)) # input(1, 1, 256, 256), output(1, 1, 256, 256)
        denoised_patch = denoised_patch.unsqueeze(0).unsqueeze(0)
        denoised_patches[i][j] = denoised_patch.cpu()
        
denoised_patches = np.uint8(denoised_patches * 255.0)

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

cv2.imwrite('test_denoising_result.png', denoised)


##########################################
# 03. PRINT RESULT
##########################################
print("[BEFORE]")
print("psnr : ", psnr_between(origin, noisy))
print("ssim : ", ssim_between(origin, noisy))

print("[AFTER]")
print("psnr : ", psnr_between(origin, denoised))
print("ssim : ", ssim_between(origin, denoised))