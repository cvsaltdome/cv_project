
import numpy as np

def get_result(real_image, calculated_image):

    h, w = np.shape(real_image)

    value1 = 0
    value2 = 0
    value3 = 0
    value4 = 0

    for i in range(0, h):
        for j in range(0, w):
            real_pixel = real_image[i, j]
            calculated_pixel = calculated_image[i, j]
            if real_pixel == 1 and calculated_pixel == 1:
                value1 += 1
            elif real_pixel == 0 and calculated_pixel == 1:
                value2 += 1
            elif real_pixel == 1 and calculated_pixel == 0:
                value3 += 1
            else
                value4 += 1
    answer = np.reshape(np.array([value1, value2, value3, value4]), (2, 2))
    return answer

