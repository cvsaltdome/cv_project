import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata

import covariance
import edge
import gl
import smootheness
import mlp


def load_network_state_dict(network, state_dict):
    network_dict = network.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in network_dict}
    network_dict.update(state_dict)
    network.load_state_dict(network_dict)


def show_in_plot(imgs):
    plt.figure()
    for i in range(11):
        plt.subplot(3, 4, i + 1)
        plt.imshow(imgs[:, :, i], cmap='gray')
    plt.show()


def load():
    tag = "mlp"
    is_cuda_available = True
    device = torch.device("cuda:0" if is_cuda_available else "cpu")

    epoch = 100
    node = 16
    drop = 0.5

    state = torch.load(os.path.join("mlp_dataset", "network", tag + "_" + str(epoch).zfill(4)), map_location=lambda storage, loc: storage)

    network = mlp.MLPNetwork(node, drop=drop).to(device).double().eval()
    load_network_state_dict(network, state["network"])

    data_dir = os.path.join("mlp_dataset", "valid")
    x_dir = os.path.join(data_dir, "x")
    img_path = os.path.join(x_dir, "183.png")

    x_raw = np.expand_dims(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY), axis=2)
    print(x_raw.shape)
    x_raw = np.append(x_raw, np.expand_dims(covariance.treat_covariance_with_multiple_window(img_path), axis=2), axis=2)
    print(x_raw.shape)
    x_raw = np.append(x_raw, np.expand_dims(edge.treat_edge_with_multiple_window(img_path), axis=2), axis=2)
    print(x_raw.shape)
    x_raw = np.append(x_raw, np.expand_dims(smootheness.treat_smoothness(img_path), axis=2), axis=2)
    print(x_raw.shape)
    x_raw = np.append(x_raw, gl.treat_all_glcm_with_multiple_window(img_path), axis=2)
    print(x_raw.shape)

    x_max = np.load(os.path.join("mlp_dataset", 'x_max.npy'))
    x_min = np.load(os.path.join("mlp_dataset", 'x_min.npy'))
    # x_max = np.zeros(9)
    # x_min = np.zeros(9)
    # for i in range(9):
    #     x_max[i] = x_raw[:, :, i + 1].max()
    #     x_min[i] = x_raw[:, :, i + 1].min()

    y_raw = np.zeros((x_raw.shape[0], x_raw.shape[1]))
    for ir in range(x_raw.shape[0]):
        x_row = x_raw[ir, :, 1: 10]
        for ib in range(x_row.shape[0]):
            x_row[ib] = 2 * (x_row[ib] - x_min) / (x_max - x_min) - 1
        print(x_row.max(), x_row.min())
        x = torch.from_numpy(x_row).to(device)
        y = network(x)
        y_row = y.cpu().detach().numpy()
        y_raw[ir, :] = 1 - y_row[:, 1]
    x_raw = np.append(x_raw, np.expand_dims(y_raw, axis=2), axis=2)
    show_in_plot(x_raw)


if __name__ == "__main__":
    load()
