import numpy as np
import math
from skimage.metrics import structural_similarity as compare_ssim
import math
import gc
import tensorflow as tf
import os
import cv2


##########################################
# 01. DENOISING METRICS
##########################################
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


##########################################
# 02. DATALOADERS
##########################################
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
            

def to_sequence(_Patches):
    seq_patches = [_Patches[0]]

    for i in range(1, len(_Patches)):
        seq_patches.append(_Patches[i])
        print("to_sequence file: [", i , "] progress: ", i ,end='                                                       \r ')

    seq_patches = np.concatenate(seq_patches, axis=0)
    return seq_patches


##########################################
# 03. PATCHING AND MERGING
##########################################
def divide_patches(image, patch_size=256, overlap_size=4):
    """
    DIVIDE IMAGES INTO PATHCES
    THIS PATCEHS MUST HAVE DUPLICARTED AREA
    
    [INPUT]
    image: (H, W, C) : numpy array
    patch_size : N : default = 256
    overlap_size : duplication factor : default = 4
    
    [OUTPUT]
    patches: (X, Y, PH, PW, C) : numpy array
    """
    
    # Calculate the step size for each axis
    step_size = patch_size - overlap_size
    
    # Calculate the number of patches along each axis
    x_axis_patches = np.ceil((image.shape[0] - overlap_size) / step_size).astype(int)
    y_axis_patches = np.ceil((image.shape[1] - overlap_size) / step_size).astype(int)
    
    # Calculate the size of the padded image
    padded_image_height = (x_axis_patches - 1) * step_size + patch_size
    padded_image_width = (y_axis_patches - 1) * step_size + patch_size
    
    # Initialize the padded image array with mirror padding
    padded_image = np.pad(image, ((0, padded_image_width), (0, padded_image_height), (0, 0)), mode='reflect')
    print("mode is mirror")
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


def concat_h(img1, img2, overlap_size=4):
    for x in range(0, img1.shape[0]):
        for y in range(0, img1.shape[1]):
            if y > img1.shape[1]-overlap_size:
                img1[x][y] = img1[x][y] * (-(y-img1.shape[1]-1)/overlap_size)
                
    for x in range(0, len(img2)):
        for y in range(0, len(img2)):
            if y < overlap_size:
                img2[x][y] = img2[x][y] * ((y)/overlap_size) 
                
    img3 = np.zeros((img2.shape[0], img1.shape[1]+img2.shape[0]-overlap_size), np.uint8)
    for x in range(0, img2.shape[0]):
        for y in range(0, img1.shape[1]+img2.shape[0]-overlap_size):
            if y < img1.shape[1]:
                img3[x][y] += img1[x][y]
                
            if y > (img1.shape[1]-overlap_size):
                img3[x][y] = np.clip(img3[x][y] + img2[x][y - (img1.shape[1] - overlap_size)], 0.0, 255.0)
          
    return img3

        
def concat_v(img1, img2, overlap_size=4):
    for x in range(0, img1.shape[0]):
        for y in range(0, img1.shape[1]):
            if x > img1.shape[0] - overlap_size:
                img1[x][y] = img1[x][y] * (-(x - img1.shape[0]-1) / overlap_size) ##

    for x in range(0, len(img2)):
        for y in range(0, len(img2[0])):
            if x < overlap_size:
                img2[x][y] = img2[x][y] * (x / overlap_size)

    img3 = np.zeros((img1.shape[0] + img2.shape[0] - overlap_size, img1.shape[1]), np.uint8)
    for x in range(0, img1.shape[0] + img2.shape[0] - overlap_size):
        for y in range(0, img1.shape[1]):
            if x < img1.shape[0]:
                img3[x][y] += img1[x][y]

            if x > (img1.shape[0] - overlap_size):
                img3[x][y] = np.clip(img3[x][y] + img2[x - (img1.shape[0] - overlap_size)][y], 0.0, 255.0)
    return img3


def merge_patches(patches, overlap_size=4):
    horizontal_concatenated = []
    for i in range(patches.shape[0]):
        row = patches[i][0]
        for j in range(1, patches.shape[1]):
            row = concat_h(row, patches[i][j], overlap_size=4)
        horizontal_concatenated.append(row)

    final_image = horizontal_concatenated[0]
    for i in range(1, len(horizontal_concatenated)):
        final_image = concat_v(final_image, horizontal_concatenated[i], overlap_size=4)
        
    return final_image


##########################################
# 04. USING FOLDER
##########################################
def img_to_patches(img, patch_size):
  patches = divide_patches(img, patch_size, overlap_size=4)
  return patches


def folder_to_patches(folder_path, patch_size, step, size):
  patches = []

  i=1
  for filename in os.listdir(folder_path):
    i+=1
    # test
    if i < size*step:
      continue
    
    img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
    this_patches = img_to_patches(img, patch_size)
    patches.extend(this_patches)
    del img, this_patches
    gc.collect()
    #print("file: [", folder_path, filename, "] progress: ", i ,end='                                                       \r ')
    
    if i > size*(step+1):
      break

  return patches


##########################################
# TEST FUNCTIONS
##########################################
if __name__ == "__main__":
    test_img = cv2.imread('./data/test/origin/Chest1.png')
    
    patches = divide_patches(test_img)
    
    cv2.imshow("patches[0][0]", patches[0][0])
    cv2.waitKey(0)
    
    merged_img = merge_patches(patches)
    
    cv2.imshow("merged_img", merged_img)
    cv2.waitKey(0)