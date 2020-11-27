
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from collections import deque
from random import *
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


def normalize(graph):
    return (graph - np.min(graph)) / (np.ptp(graph))


def convert_edge_to_normal(edge_region):
    normalized_edge = normalize(edge_region)
    th_lo = 0.1
    th_hi = 0.2
    hyst = apply_hysteresis_threshold(normalized_edge, th_lo, th_hi)
    return hyst
def flood_fill(edge_region, covariance_result):
    h, w = np.shape(edge_region)

    graph = convert_edge_to_normal(edge_region)
    normalize_covariance = normalize(covariance_result)

    number_of_area = 0
    flood_image = np.zeros((h, w))
    result_image = np.zeros((h, w))

    flood_fill_data = []

    for x in range(0, h):
        for y in range(0, w):
            if flood_image[x, y] == 0:
                number_of_area = number_of_area + 1
                result = bfs(x, y, h, w, graph, flood_image, number_of_area, normalize_covariance)
                flood_fill_data.append([x, y, result])
    print(flood_fill_data)
    for x, y, result in flood_fill_data:
        if result > 0.3:
            bfs_for_result(x, y, h, w, result_image, flood_image)

    return flood_image, result_image

