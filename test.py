import os
import shutil
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np

task = 'Real_Denoising'

################################
# 01. PREPARE MODEL
################################


def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters


# Get model weights and parameters
parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
weights, parameters = get_weights_and_parameters(task, parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)
model

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

################################
# 02. INFERENCE
################################

input_dir = 'C:/Users/NDT/Desktop/Image_denoising/Data/test_patches/gaussian/Chest1(3072_3072)_IP.dcm.png'
out_dir = 'oneimageresult'

os.makedirs(out_dir, exist_ok=True)
extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
files = natsorted(glob(os.path.join(input_dir, '*')))

img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")
with torch.no_grad():
  for filepath in tqdm(files):
      # print(file_)
      #torch.cuda.ipc_collect()
      #torch.cuda.empty_cache()
      img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
      #input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()
      input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0)

      # Pad the input if not_multiple_of 8
      h,w = input_.shape[2], input_.shape[3]
      H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
      padh = H-h if h%img_multiple_of!=0 else 0
      padw = W-w if w%img_multiple_of!=0 else 0
      input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

      restored = model(input_)
      restored = torch.clamp(restored, 0, 1)

      # Unpad the output
      restored = restored[:,:,:h,:w]

      restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
      restored = img_as_ubyte(restored[0])

      filename = os.path.split(filepath)[-1]
      cv2.imwrite(os.path.join(out_dir, filename),cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))