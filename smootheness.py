import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
import imagehelper
import matplotlib.pyplot as plt


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
def show_in_plot(images):
    image_size = len(images)
    plt.figure()
    for i in range(0, image_size):
        plt.subplot(image_size, 1, i+1)
        plt.imshow(images[i], cmap='gray')
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

def treat(img_path, result_path):
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    I = cv2.resize(I, dsize=(200, 200), interpolation=cv2.INTER_AREA)
    I = I / 255

    result_image = cv2.cvtColor(cv2.imread(result), cv2.COLOR_BGR2GRAY)
    result_image = cv2.resize(result_image, dsize=(200, 200), interpolation=cv2.INTER_AREA)

    patch_size = 25
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=patch_size)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=patch_size)
    G = np.sqrt(Gx * Gx + Gy * Gy)
    G = G / np.max(G)
    """
        normalize
    """

    """
    smootheness window size
    """

    smoothness_window_size = 5
    w, h = I.shape
    smoothness_window = np.zeros(I.shape)
    
    for i in range(0, w):
        for j in range(0, h):
            gradient_patch = get_patch_at(Gx, i, j, smoothness_window_size)
            sum = np.sum(gradient_patch)
            smoothness_window[i, j] = sum

    smoothness_window = smoothness_window / np.max(smoothness_window)
    show_in_plot([I, smoothness_window, result_image])


images_path = imagehelper.get_image_files()
image_pairs = imagehelper.get_image_pairs()
for image, result in image_pairs:
    treat(image, result)
