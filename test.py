import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import covariance
import edge
import gl
import smootheness


def get_image_files():
    return glob.glob(os.path.join("data/original", '6.png'))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
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


def main(img_path):
    chaos = covariance.treat_covariance_with_multiple_window(img_path)
    e = edge.treat_edge_with_multiple_window_normalized(img_path)
    GLCM = gl.treat_glcm(img_path, 3, 'homogeneity')
    sm = smootheness.treat_smoothness(img_path, 3, 5)
    show_in_plot(chaos, sm, e, GLCM)
    print(np.max(e), np.min(e), np.max(GLCM), np.min(GLCM), np.min(sm))


if __name__ == "__main__":
    images_path = get_image_files()
    for image in images_path:
        main(image)
