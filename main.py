# %%

import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
import imagehelper
import matplotlib.pyplot as plt
import glcm
import covariance
import smootheness
import edge

image_pairs = imagehelper.get_image_pairs()
for image, result in image_pairs:
    I = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    R = cv2.cvtColor(cv2.imread(result), cv2.COLOR_BGR2GRAY)
    # glcm
    non_salt_patches, salt_patches = glcm.treat_glcm(image, result)
    # covariance
    covariance_result = covariance.treat_covariance(image)
    # edge_result
    edge_result = edge.treat_edge(image)
    # smootheness
    smootheness_result = smootheness.treat_result(image)

    imagehelper.show_in_plot([
        I, R,
        covariance_result, edge_result, smootheness_result])