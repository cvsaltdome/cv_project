# %%

import cv2

import covariance
import edge
import gl
import imagehelper
import smootheness
import floodfill

image_pairs = imagehelper.get_image_pairs()

if __name__ == "__main__":
    for image, result in image_pairs:
        I = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
        R = cv2.cvtColor(cv2.imread(result), cv2.COLOR_BGR2GRAY)
        # glcm
        # non_salt_patches, salt_patches = glcm.treat_glcm(image, result)
        # covariance
        covariance_result = covariance.treat_covariance_with_multiple_window_normalized(image)
        # edge_result
        edge_result = edge.treat_edge_with_multiple_window(image)
        # GL
        # gl_result=gl.treat_glcm_with_multiple_window_normalized(image,10,'homogeneity')
        # # smootheness
        # smootheness_result = smootheness.treat_smoothness_normalized(image)
        # imagehelper.show_in_plot([
        #     I, R,
        #     covariance_result, edge_result, smootheness_result, gl_result
        # ])

        graph = floodfill.convert_edge_to_normal(edge_result)

        floodfill_result, colored_results = floodfill.flood_fill(edge_result, covariance_result)
        imagehelper.show_in_plot([
            I,
            covariance_result, edge_result
        ] + colored_results + [R])
        #
        # for covariance_w in np.linspace(0, 1.0, num=5):
        #     for edge_w in np.linspace(0, 1.0, num = 5):
        #         for smootheness_w in np.linspace(0, 1, num = 5):
        #             for covariance_p in np.linspace(1, 2.0, num=5):
        #                 for edge_p in np.linspace(1, 2.0, num=5):
        #                     for smootheness_p in np.linspace(1, 2, num=5):
        #
        #                         final_result = np.power(covariance_result, covariance_p) * covariance_w + np.power(edge_result, edge_p) * edge_w + np.power(smootheness_result, smootheness_p) * smootheness_w
        #
        #                         final_result = final_result / np.max(final_result)
        #                         imagehelper.show_in_plot([
        #                             I, R,
        #                             covariance_result, edge_result, smootheness_result,
        #                             final_result
        #                         ])
        #
        #
