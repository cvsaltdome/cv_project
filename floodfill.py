
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from collections import deque
from random import *
import imagehelper
from skimage.filters import apply_hysteresis_threshold


def is_in_area(x, y, h, w):
    return 0 <= x < h and 0 <= y < w

def bfs(x, y, h, w, graph, flood_image, number_of_area, covariance_image):
    queue = deque([])
    dxs = [-1, 0, 1, 0]
    dys = [0, 1, 0, -1]

    color = graph[x, y]

    queue.appendleft([x, y])
    flood_image[x, y] = number_of_area
    number_of_flood_pixel = 1
    number_of_result_sum = covariance_image[x, y]
    while queue:
        x, y = queue.pop()
        for dx, dy in zip(dxs, dys):
            next_x = x + dx
            next_y = y + dy
            if is_in_area(next_x, next_y, h, w):
                if graph[next_x, next_y] == color and flood_image[next_x, next_y] == 0:
                    flood_image[next_x, next_y] = number_of_area
                    queue.appendleft([next_x, next_y])
                    number_of_flood_pixel = number_of_flood_pixel + 1
                    number_of_result_sum += covariance_image[next_x, next_y]

    result_average = number_of_result_sum / number_of_flood_pixel
    return result_average

def bfs_for_result(x, y, h, w, result_image, flood_image):

    queue = deque([])
    dxs = [-1, 0, 1, 0]
    dys = [0, 1, 0, -1]

    color = flood_image[x, y]
    queue.appendleft([x, y])
    result_image[x, y] = 1
    while queue:
        x, y = queue.pop()
        for dx, dy in zip(dxs, dys):
            next_x = x + dx
            next_y = y + dy
            if is_in_area(next_x, next_y, h, w):
                if flood_image[next_x, next_y] == color and result_image[next_x, next_y] == 0:
                    result_image[next_x, next_y] = 1
                    queue.appendleft([next_x, next_y])

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

def nms(image):
    h, w = np.shape(image)
    patch_size = int(np.round(min(h, w) / 20))

    if patch_size % 2 == 0:
        patch_size += 1


    result = np.zeros((h, w))

    for i in range(0, h):
        for j in range(0, w):
            patch = get_patch_at(image, i, j, patch_size)
            patch_min = np.max(patch)
            if image[i, j] == patch_min:
                result[i, j] = 1
            else:
                result[i, j] = 0
    return result

def normalize(graph):
    return (graph - np.min(graph)) / (np.ptp(graph))


def convert_edge_to_normal(edge_region):
    normalized_edge = normalize(edge_region)
    th_lo = 0.7
    th_hi = 0.8
    hyst = apply_hysteresis_threshold(normalized_edge, th_lo, th_hi)
    return hyst

def convert_to_image(graph):
    normalized_edge = normalize(graph) * 255
    return normalized_edge.astype(np.uint8)

def contour(edge_region):
    image = convert_to_image(edge_region)

    new_image = np.zeros(np.shape(edge_region))
    for i in range(0, 255, 32):
        ret, thr = cv2.threshold(image, i, 255, 0)
        new_image += thr
    return new_image


def flood_fill(edge_region, covariance_result):
    h, w = np.shape(edge_region)

    graph = contour(edge_region)
    normalize_covariance = normalize(covariance_result)

    number_of_area = 0
    flood_image = np.zeros((h, w))

    flood_fill_data = []

    for x in range(0, h):
        for y in range(0, w):
            if flood_image[x, y] == 0:
                number_of_area = number_of_area + 1
                result = bfs(x, y, h, w, graph, flood_image, number_of_area, normalize_covariance)
                flood_fill_data.append([x, y, result])
    print(flood_fill_data)

    eval = []
    for threshold in range(1, 10):
        result_image = np.zeros((h, w))
        for x, y, result in flood_fill_data:
            if result > (threshold / 10):
                bfs_for_result(x, y, h, w, result_image, flood_image)
        eval.append(result_image)



    return flood_image, eval

