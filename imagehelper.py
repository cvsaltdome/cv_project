import shutil
import glob, os
parent_dir = r"data"
image_original = []
image_result = []
import cv2
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

def get_image_files():
    return glob.glob(os.path.join("data/original", '*.png'))


def get_images():
    cv_images = []
    for image_file in glob.glob(os.path.join("data/original", '*.png')):
        cv_image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY)
        cv_images.append(cv_image)
    return cv_images

