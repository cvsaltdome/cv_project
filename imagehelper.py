import shutil
import glob, os
parent_dir = r"data"
image_original = []
image_result = []
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

# for image_file in glob.glob(os.path.join(parent_dir, '*.png')):
#     if str(image_file).endswith("- 복사본.png"):
#         image_result.append(str(image_file))
#     else:
#         image_original.append(str(image_file))
# print(image_original)
# print(image_result)
#
# count = 0
# for original in image_original:
#     result = ""
#     for result_candidate in image_result:
#         if result_candidate.startswith(original[:-4]):
#             print(result_candidate)
#             print(original)
#             shutil.move(original, f"data/original/{count}.png")
#             shutil.move(result_candidate, f"data/result/{count}.png")
#
#             count += 1
#             break

"""
이미지를 전부 가져온다.
"""

def get_image_pairs():
    original = []
    result = []
    for i in range(0, 15):
        original.append(f"data/original/{i}.png")
        result.append(f"data/result/{i}.png")
    return zip(original, result)

def get_image_files():
    return glob.glob(os.path.join("data/original", '*.png'))

def get_image_mask():
    return glob.glob(os.path.join("data/result", '*.png'))

def get_images():
    cv_images = []
    for image_file in glob.glob(os.path.join("data/original", '*.png')):
        cv_image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY)
        cv_images.append(cv_image)
    return cv_images

"""
플롯을 보여 준다.
"""
def show_in_plot(images):
    image_size = len(images)
    plt.figure()
    for i in range(0, image_size):
        plt.subplot(image_size, 1, i+1)
        plt.imshow(images[i], cmap='gray')
    plt.show()
