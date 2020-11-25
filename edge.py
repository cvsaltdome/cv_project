import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
import imagehelper
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle



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


def treat_edge(img_path, sobel_patch_size = 25, gradient_path_size = 25):
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    I = I / 255
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=sobel_patch_size)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=sobel_patch_size)
    G = np.sqrt(Gx * Gx + Gy * Gy)

    w, h = I.shape
    edge = np.zeros((w, h))

    for i in range(0, w):
        for j in range(0, h):
            g_patch = get_patch_at(G, i, j, gradient_path_size)
            edge[i, j] = np.max(g_patch) - np.min(g_patch)
    return edge


def treat_edge_with_multple_window(img_path):
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    sum = np.zeros(I.shape)
    for patch_size in range(5, 11, 2):
        print(patch_size)
        sum += treat_edge(img_path, 3, patch_size)
    average = sum / np.sum(sum)
    tv_denoised = denoise_tv_chambolle(average, weight=10)
    return tv_denoised