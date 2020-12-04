import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as tdata

import covariance
import edge
import gl
import mlp
import smootheness


# from scipy import ndimage


def load_network_state_dict(network, state_dict):
    network_dict = network.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in network_dict}
    network_dict.update(state_dict)
    network.load_state_dict(network_dict)


def show_in_plot(imgs):
    plt.figure()
    for i in range(imgs.shape[2]):
        plt.subplot(3, 4, i + 1)
        if i != 10:
            plt.imshow(imgs[:, :, i], cmap='gray')
        else:
            plt.imshow(imgs[:, :, i], cmap='gray', vmin=0, vmax=1)
    plt.show()


def load():
    data_usage = "test"
    img_name = "1.png"

    tag = "mlp"
    is_cuda_available = True
    device = torch.device("cuda:0" if is_cuda_available else "cpu")

    epoch = 94
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
    tmp = np.expand_dims(covariance.treat_covariance_with_multiple_window(img_path), axis=2)
    # tmp = ndimage.gaussian_filter(tmp, 2, mode="nearest")
    data = np.append(data, tmp, axis=2)
    print(data.shape)
    tmp = np.expand_dims(edge.treat_edge_with_multiple_window(img_path), axis=2)
    # tmp = ndimage.gaussian_filter(tmp, 2, mode="nearest")
    data = np.append(data, tmp, axis=2)
    print(data.shape)
    tmp = np.expand_dims(smootheness.treat_smoothness(img_path), axis=2)
    # tmp = ndimage.gaussian_filter(tmp, 2, mode="nearest")
    data = np.append(data, tmp, axis=2)
    print(data.shape)
    tmp = gl.treat_all_glcm_with_multiple_window(img_path)
    # for i in range(tmp.shape[2]):
    #     tmp[:, :, i] = ndimage.gaussian_filter(tmp[:, :, i], 2, mode="nearest")
    data = np.append(data, tmp, axis=2)
    print(data.shape)

    x_max = np.load(os.path.join("mlp_dataset", 'x_max.npy'))
    x_min = np.load(os.path.join("mlp_dataset", 'x_min.npy'))

    # x_max = np.max(data[:, :, 1: 10], axis=(0, 1))
    # # x_max[5] /= 0.95
    # # x_max[6] /= 0.95
    # # x_max[8] /= 0.95
    # x_min = np.min(data[:, :, 1: 10], axis=(0, 1))

    y_raw = np.zeros((data.shape[0], data.shape[1]))
    for ir in range(data.shape[0]):
        x_row = data[ir, :, 1: 10].copy()
        for ib in range(x_row.shape[0]):
            x_row[ib] = 2 * (x_row[ib] - x_min) / (x_max - x_min) - 1
        x = torch.from_numpy(x_row).to(device)
        y = network(x)
        y_row = y.cpu().detach().numpy()
        y_raw[ir, :] = 1 - y_row[:, 1]
    data = np.append(data, np.expand_dims(y_raw, axis=2), axis=2)

    data_dir = os.path.join("mlp_dataset", data_usage)
    y_dir = os.path.join(data_dir, "y")
    img_path = os.path.join(y_dir, img_name)
    data = np.append(data, np.expand_dims(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY), axis=2), axis=2)

    show_in_plot(data)


def check():
    tag = "mlp"
    epoch = 100
    state = torch.load(os.path.join("mlp_dataset", "network", tag + "_" + str(epoch).zfill(4)), map_location=lambda storage, loc: storage)

    train_losses = state["train_losses"]
    valid_losses = state["valid_losses"]
    accuracies = state["accuracies"]

    for i in range(100):
        print(i + 1, train_losses[i], valid_losses[i], accuracies[i])

    vl = np.array(valid_losses)
    ac = np.array(accuracies)

    print(vl.argmin(), ac.argmax())


def cal_confusion_matrix():
    cudnn.benchmark = True
    tag = "mlp"
    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda_available else "cpu")

    batch_size = 256
    epoch = 94
    node = 16
    drop = 0.5

    network = mlp.MLPNetwork(node, drop=drop).to(device).double().eval()

    train_set = mlp.SeismicAttributeDataset(data_usage="train", is_new=False)
    train_loader = tdata.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    valid_set = mlp.SeismicAttributeDataset(data_usage="valid", is_new=False)
    valid_loader = tdata.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

    state = torch.load(os.path.join("mlp_dataset", "network", tag + "_" + str(epoch).zfill(4)))

    network.load_state_dict(state["network"])

    cm = np.zeros((2, 2), dtype=np.int32)
    for i_batch, (x_data, y_data) in enumerate(train_loader):
        if x_data.size()[0] != batch_size:
            continue

        x = x_data.to(device)
        label = y_data.to(device)
        y = network(x)

        _, prediction = torch.max(y.data, dim=1)
        _, answer = torch.max(label.data, dim=1)
        for i_data in range(x.data.size()[0]):
            cm[answer[i_data], prediction[i_data]] += 1

        if (i_batch + 1) % 100 == 0:
            print("batch index: {:4}".format(i_batch + 1))
    print(cm)

    cm = np.zeros((2, 2), dtype=np.int32)
    for i_batch, (x_data, y_data) in enumerate(valid_loader):
        if x_data.size()[0] != batch_size:
            continue

        x = x_data.to(device)
        label = y_data.to(device)
        y = network(x)

        _, prediction = torch.max(y.data, dim=1)
        _, answer = torch.max(label.data, dim=1)
        for i_data in range(x.data.size()[0]):
            cm[answer[i_data], prediction[i_data]] += 1

        if (i_batch + 1) % 100 == 0:
            print("batch index: {:4}".format(i_batch + 1))
    print(cm)


if __name__ == "__main__":
    cal_confusion_matrix()
