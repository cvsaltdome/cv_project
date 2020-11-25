from skimage.feature import greycomatrix, greycoprops
from skimage import data
import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
import imagehelper
import matplotlib.pyplot as plt
import shutil
import glob, os

from skimage.restoration import denoise_tv_chambolle
parent_dir = r"data"
image_original = []
image_result = []
import cv2
import numpy as np
import random


def get_image_files():
    return glob.glob(os.path.join("data/original", '*.png'))


def show_in_plot(img1, img2, img3, img4):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(img3, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(img4, cmap='gray')
    plt.show()


def get_patch_at(pixel_grid, i, j, size):
    x_length, y_length = pixel_grid.shape
    half_size = int(size / 2)
    start_x: int = max(0, i - half_size)
    end_x: int = min(x_length, i + half_size + 1)
    start_y: int = max(0, j - half_size)
    end_y: int = min(y_length, j + half_size + 1)
    pad_start_x: int = max(0, -(i - half_size))
    pad_end_x: int = max(0, (i + half_size + 1) - x_length)
    pad_start_y: int = max(0, -(j - half_size))
    pad_end_y: int = max(0, (j + half_size + 1) - y_length)
    pad_value = ((pad_start_x, pad_end_x), (pad_start_y, pad_end_y))
    sliced = pixel_grid[start_x:end_x, start_y:end_y]
    if pad_start_x == 0 and pad_end_x == 0 and pad_start_y == 0 and pad_end_y == 0:
        return sliced
    else:
        return np.pad(sliced, pad_value, 'edge')


def get_cov(x, y, patch_size):
    x_avg = np.average(x)
    y_avg = np.average(y)
    sum = np.multiply(x - x_avg, y - y_avg)
    return np.sum(sum) / patch_size * patch_size


def treat_glcm(img_path, patch_size = 5):
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    w, h = I.shape
    GLCM = np.zeros((w, h))

    for i in range(0, w):
        for j in range(0, h):
            patch = get_patch_at(I, i, j, patch_size)
            glcm = greycomatrix(patch, distances=[1], angles=[90], levels=256, symmetric=True, normed=True)
            GLCM[i, j] = greycoprops(glcm, 'correlation')[0, 0]
    result = (GLCM - np.min(GLCM)) / (np.ptp(GLCM))
    return result

def treat_glcm_with_multple_window(img_path):
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    sum = np.zeros(I.shape)
    for patch_size in range(3, 11, 12):
        print(patch_size)
        sum += treat_glcm(img_path, patch_size)
    average = sum / np.sum(sum)
    tv_denoised = denoise_tv_chambolle(average, weight=10)
    return tv_denoised
