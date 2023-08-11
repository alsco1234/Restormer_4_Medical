# import packages
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2

import torch

from basicsr.models.archs.restormer_arch import Restormer
from pdb import set_trace as stx
import math
from skimage.metrics import structural_similarity as compare_ssim

#import matplotlib.pyplot as plt
import gc
import tensorflow as tf

class Dataloder(tf.keras.utils.Sequence):
    def __init__(self, X,y,batch_size=1, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(X))

    def __getitem__(self, i):
        # collect batch data
        batch_x = self.X[i * self.batch_size : (i+1) * self.batch_size]
        batch_y = self.y[i * self.batch_size : (i+1) * self.batch_size]

        return tuple((batch_x,batch_y))

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# image to patches
def img_to_patches(img, patch_size):
  patches = divide_patches(img, patch_size, overlap_size=4)
  return patches

# images folder  to patches
def folder_to_patches(folder_path, patch_size, step, size):
  patches = []

  i=1
  for filename in os.listdir(folder_path):
    i+=1
    # test
    if i < size*step:
      continue
    
    img = cv2.imread(os.path.join(folder_path, filename))
    this_patches = img_to_patches(img, patch_size)
    patches.extend(this_patches)
    del img, this_patches
    gc.collect()
    #print("file: [", folder_path, filename, "] progress: ", i ,end='                                                       \r ')
    
    if i > size*(step+1):
      break

  return patches

def divide_patches(image, patch_size=256, overlap_size=4):
    
    # Calculate the step size for each axis
    step_size = patch_size - overlap_size
    
    # Calculate the number of patches along each axis
    x_axis_patches = np.ceil((image.shape[0] - overlap_size) / step_size).astype(int)
    y_axis_patches = np.ceil((image.shape[1] - overlap_size) / step_size).astype(int)
    
    # Calculate the size of the padded image
    padded_image_height = (x_axis_patches - 1) * step_size + patch_size
    padded_image_width = (y_axis_patches - 1) * step_size + patch_size
    
    # Initialize the padded image array
    padded_image = np.zeros((padded_image_height, padded_image_width, image.shape[2]))
    
    # Copy the input image into the padded image array
    padded_image[:image.shape[0], :image.shape[1]] = image
    
    # Initialize the patches array
    patches = np.zeros((x_axis_patches, y_axis_patches, patch_size, patch_size, padded_image.shape[2]))
    
    # Iterate over the patches and extract them from the padded image
    for i in range(x_axis_patches):
        for j in range(y_axis_patches):
            x_start = i * step_size
            y_start = j * step_size
            
            patches[i, j] = padded_image[x_start:x_start+patch_size, y_start:y_start+patch_size]
    
    return patches


def merge_patches(patches, origin_H, origin_W, overlap_size=4):
    # Calculate the step size for each axis
    step_size = patches.shape[2] - overlap_size # 252
    
    # Calculate the size of the merged image
    padded_image_height = (patches.shape[0] - 1) * step_size + patches.shape[2]
    padded_image_width = (patches.shape[1] - 1) * step_size + patches.shape[3]
    
    # Initialize the padded merged image array
    padded_image = np.zeros((padded_image_height, padded_image_width, patches.shape[4]))
    
    # Iterate over the patches and add them to the padded merged image
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            x_start = i * patches.shape[2] - i * overlap_size
            y_start = j * patches.shape[3] - j * overlap_size
            
            # Add the patch to the padded merged image
            for oi in range(patches.shape[2]):
                for oj in range(patches.shape[3]):
                    # already exist => overlap point. let's average
                    if padded_image[x_start + oi, y_start + oj, 0]!= 0:
                        padded_image[x_start + oi, y_start + oj] = (padded_image[x_start + oi, y_start + oj] + patches[i, j][oi, oj]) / 2.0
                    else:
                        padded_image[x_start + oi, y_start + oj] = patches[i, j][oi, oj]

    # Crop the padded merged image to the original shape of the input image
    image = padded_image[:origin_H, :origin_W]
    
    return image

# Learning 위해 sequence하게 만들기
def to_sequence(_Patches):
    seq_patches = [_Patches[0]]

    for i in range(1, len(_Patches)):
        seq_patches.append(_Patches[i])
        print("to_sequence file: [", i , "] progress: ", i ,end='                                                       \r ')

    seq_patches = np.concatenate(seq_patches, axis=0)
    return seq_patches