import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as compare_ssim

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

noisy = cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test/gaussian/Chest1(3072_3072)_IP.dcm.png', cv2.IMREAD_GRAYSCALE)
print(noisy)
origin = cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Data/test/origin/Chest1(3072_3072)_IP.dcm.png', cv2.IMREAD_GRAYSCALE)
denoised = cv2.imread('C:/Users/NDT/Desktop/Image_denoising/Chest1_restormer11_merging_85.png', cv2.IMREAD_GRAYSCALE)
'''
# bit수준 24 => 8로 맞추기. 이때도 같은 방법 사용
denoised_8 = denoised - denoised.min()
denoised_8 = denoised_8 / denoised.max()
denoised_8 = denoised_8 * 255.0
denoised_8 = [np.rint(x) for x in denoised_8]

denoised_8 = np.array(denoised_8)

noisy = noisy.astype(np.uint8)
origin = origin.astype(np.uint8)
denoised_8 = denoised_8.astype(np.uint8)

print(max(noisy[0]), "and ", max(origin[0]), "amd", max(denoised[0]))
print(max(noisy[1]), "and ", max(origin[1]), "amd", max(denoised[1]))
print(max(noisy[2]), "and ", max(origin[2]), "amd", max(denoised[2]))
print(max(noisy[3]), "and ", max(origin[3]), "amd", max(denoised[3]))
print(max(noisy[4]), "and ", max(origin[4]), "amd", max(denoised[0]))
'''

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