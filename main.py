# %%

import cv2

import covariance
import edge
import gl
import glcm
import imagehelper
import smootheness
import floodfill

image_pairs = imagehelper.get_image_pairs()
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
            else:
                value4 += 1
    answer = np.array([value1, value2, value3, value4]).astype(int)
    return answer

if __name__ == "__main__":

    data = []
    for i in range(0, 10):
        zero = np.zeros(4).astype(int)
        data.append(zero)

    count = 0

    for image, result in image_pairs:
        I = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
        R = cv2.cvtColor(cv2.imread(result), cv2.COLOR_BGR2GRAY)
        # glcm
        # non_salt_patches, salt_patches = glcm.treat_glcm(image, result)
        # covariance
        # covariance_result = covariance.treat_covariance_with_multiple_window_normalized(image)
        # edge_result
        edge_result = edge.treat_edge_with_multiple_window(image)
        # GL
        gl_result=gl.treat_glcm_with_multiple_window_normalized(image,10,'homogeneity')
        # # smootheness
        # smootheness_result = smootheness.treat_smoothness_normalized(image)
        # imagehelper.show_in_plot([
        #     I, R,
        #     covariance_result, edge_result, smootheness_result, gl_result
        # ])

        graph = floodfill.convert_edge_to_normal(edge_result)
        floodfill_result, colored_results = floodfill.flood_fill(edge_result, gl_result)
        imagehelper.show_in_plot([
            I, R,
            gl_result, edge_result, graph, floodfill_result,
        ] + colored_results)
        converted_R = R / 255
        for index, colored_result in enumerate(colored_results):
            final_result = get_result(converted_R, colored_result)
            data[index] += final_result
        print(data)
        print(count)
        for index, value in enumerate(data):
            print(f"{index}")
            value1, value2, value3, value4 = value
            print("{:12} {:12} {:12} {:12}".format(value1, value2, value3, value4))
        count += 1

    for index, value in enumerate(data):
        print(f"{index}")
        value1, value2, value3, value4 = value
        print("{:12} {:12} {:12} {:12}".format(value1, value2, value3, value4))
