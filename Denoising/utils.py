import numpy as np
import math
from skimage.metrics import structural_similarity as compare_ssim
from basicsr.models.archs.restormer_arch import Restormer
from pdb import set_trace as stx
import math

#import matplotlib.pyplot as plt
import gc
import tensorflow as tf

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
    
    # 각 축에 대한 스텝 크기 계산
    step_size = patch_size - overlap_size
    
    # 각 축의 패치 수 계산
    x_axis_patches = np.ceil((image.shape[0] - overlap_size) / step_size).astype(int)
    y_axis_patches = np.ceil((image.shape[1] - overlap_size) / step_size).astype(int)
    
    # 패딩된 이미지의 크기 계산
    padded_image_height = (x_axis_patches - 1) * step_size + patch_size
    padded_image_width = (y_axis_patches - 1) * step_size + patch_size
    
    # 미러 패딩으로 패딩된 이미지 배열 초기화
    padded_image = np.pad(image, ((0, padded_image_width), (0, padded_image_height)), mode='reflect')
    
    # 입력 이미지를 패딩된 이미지 배열에 복사
    padded_image[:image.shape[0], :image.shape[1]] = image
    
    # 패치 배열 초기화
    patches = np.zeros((x_axis_patches, y_axis_patches, patch_size, patch_size))
    
    # 패치를 추출하여 패딩된 이미지에서 가져옴
    for i in range(x_axis_patches):
        for j in range(y_axis_patches):
            x_start = i * step_size
            y_start = j * step_size
            
            patches[i, j] = padded_image[x_start:x_start+patch_size, y_start:y_start+patch_size]
    
    return patches


def concat_h(img1, img2):
    for x in range(0, img1.shape[0]):
        for y in range(0, img1.shape[1]):
            if y > img1.shape[1]-85:
                img1[x][y] = img1[x][y] * (-(y-img1.shape[1]-1)/85) ## -1하는게 보기에 제일 ㄱㅊ았음
                
    for x in range(0, len(img2)):
        for y in range(0, len(img2)):
            if y < 85:
                img2[x][y] = img2[x][y] * ((y)/85) ## 예를 y+1하는게 논리적으론 맞는데.. 보기에 이상해짐 ㅠ
                

    img3 = np.zeros((256, img1.shape[1]+256-85), np.uint8)

    for x in range(0, 256):
        for y in range(0, img1.shape[1]+256-85):
            if y < img1.shape[1]:
                img3[x][y] += img1[x][y]
                
            if y > (img1.shape[1]-85):
                img3[x][y] += img2[x][y-(img1.shape[1]-85)]
                
    return img3

        
def concat_v(img1, img2):
    for x in range(0, img1.shape[0]):
        for y in range(0, img1.shape[1]):
            if x > img1.shape[0] - 85:
                img1[x][y] = img1[x][y] * (-(x - img1.shape[0]-1) / 85) ##

    for x in range(0, len(img2)):
        for y in range(0, len(img2[0])):
            if x < 85:
                img2[x][y] = img2[x][y] * (x / 85)

    img3 = np.zeros((img1.shape[0] + img2.shape[0] - 85, img1.shape[1]), np.uint8)

    for x in range(0, img1.shape[0] + img2.shape[0] - 85):
        for y in range(0, img1.shape[1]):
            if x < img1.shape[0]:
                img3[x][y] += img1[x][y]

            if x > (img1.shape[0] - 85):
                img3[x][y] += img2[x - (img1.shape[0] - 85)][y]

    return img3


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
    origin_image_path = 'C:/Users/NDT/Desktop/Image_denoising/Data/test/origin/Chest1(3072_3072)_IP.dcm.png'
    origin_image = cv2.imread(origin_image_path)
    
    patches = divide_patches(origin_image, patch_size=256, overlap_size=85)