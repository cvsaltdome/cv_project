import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage.restoration import denoise_tv_chambolle


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


def treat_glcm(img_path, patch_size=5, mod='contrast'):
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    w, h = I.shape

    GLCM = np.zeros((w, h))
    for i in range(0, w):
        for j in range(0, h):
            patch = get_patch_at(I, i, j, patch_size)
            glcm = greycomatrix(patch, distances=[1], angles=[90], levels=256, symmetric=True, normed=True)
            GLCM[i, j] = greycoprops(glcm, mod)[0, 0]
    return GLCM


def treat_glcm_with_multiple_window(img_path, weight=10):
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

    summation = np.zeros(I.shape)
    for patch_size in range(3, 11, 12):
        print(patch_size)
        summation += treat_glcm(img_path, patch_size, 'correlation')
    average = summation / np.sum(summation)
    tv_denoised = denoise_tv_chambolle(average, weight=weight)
    return tv_denoised


def treat_glcm_normalized(img_path, patch_size=5, mod='contrast'):
    GLCM = treat_glcm(img_path, patch_size, mod)
    result = (GLCM - np.min(GLCM)) / (np.ptp(GLCM))
    return result


def treat_glcm_with_multiple_window_normalized(img_path, weight=10):
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

    summation = np.zeros(I.shape)
    for patch_size in range(3, 11, 12):
        print(patch_size)
        summation += treat_glcm_normalized(img_path, patch_size, 'correlation')
    average = summation / np.sum(summation)
    tv_denoised = denoise_tv_chambolle(average, weight=weight)
    return tv_denoised
