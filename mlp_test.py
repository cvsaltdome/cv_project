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
        if i != 10:
            plt.imshow(imgs[:, :, i], cmap='gray')
        else:
            plt.imshow(imgs[:, :, i], cmap='gray', vmin=0, vmax=1)
    plt.show()


def load():
    data_usage = "test"
    img_name = "11.png"

    tag = "mlp"
    is_cuda_available = True
    device = torch.device("cuda:0" if is_cuda_available else "cpu")

    epoch = 85
    node = 16
    drop = 0.5

    state = torch.load(os.path.join("mlp_dataset", "network", tag + "_" + str(epoch).zfill(4)), map_location=lambda storage, loc: storage)

    network = mlp.MLPNetwork(node, drop=drop).to(device).double().eval()
    load_network_state_dict(network, state["network"])

    data_dir = os.path.join("mlp_dataset", data_usage)
    x_dir = os.path.join(data_dir, "x")
    img_path = os.path.join(x_dir, img_name)

    data = np.expand_dims(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY), axis=2)
    print(data.shape)
    data = np.append(data, np.expand_dims(covariance.treat_covariance_with_multiple_window(img_path), axis=2), axis=2)
    print(data.shape)
    data = np.append(data, np.expand_dims(edge.treat_edge_with_multiple_window(img_path), axis=2), axis=2)
    print(data.shape)
    data = np.append(data, np.expand_dims(smootheness.treat_smoothness(img_path), axis=2), axis=2)
    print(data.shape)
    data = np.append(data, gl.treat_all_glcm_with_multiple_window(img_path), axis=2)
    print(data.shape)

    x_max = np.load(os.path.join("mlp_dataset", 'x_max.npy'))
    x_min = np.load(os.path.join("mlp_dataset", 'x_min.npy'))

    y_raw = np.zeros((data.shape[0], data.shape[1]))
    for ir in range(data.shape[0]):
        x_row = data[ir, :, 1: 10]
        for ib in range(x_row.shape[0]):
            x_row[ib] = 2 * (x_row[ib] - x_min) / (x_max - x_min) - 1
        x = torch.from_numpy(x_row).to(device)
        y = network(x)
        y_row = y.cpu().detach().numpy()
        y_raw[ir, :] = 1 - y_row[:, 1]
    data = np.append(data, np.expand_dims(y_raw, axis=2), axis=2)

    # data_dir = os.path.join("mlp_dataset", data_usage)
    # y_dir = os.path.join(data_dir, "y")
    # img_path = os.path.join(y_dir, img_name)
    # data = np.append(data, np.expand_dims(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY), axis=2), axis=2)

    show_in_plot(data)


def check():
    tag = "mlp"
    epoch = 100
    state = torch.load(os.path.join("mlp_dataset", "network", tag + "_" + str(epoch).zfill(4)), map_location=lambda storage, loc: storage)

    valid_losses = state["valid_losses"]
    accuracies = state["accuracies"]

    a = np.array(valid_losses)
    i = np.argmin(a)

    print(valid_losses)
    print(i)

    a = np.array(accuracies)
    i = np.argmax(a)

    print(accuracies)
    print(i)


if __name__ == "__main__":
    load()
