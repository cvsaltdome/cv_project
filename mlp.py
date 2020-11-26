import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata

import covariance
import edge
import gl
import smootheness


class SeismicAttributeDataset(tdata.Dataset):
    def __init__(self, is_test, is_new):
        if is_test:
            data_dir = os.path.join("", "mlp_dataset", "test")
        else:
            data_dir = os.path.join("", "mlp_dataset", "valid")

        if is_new:
            x_dir = os.path.join(data_dir, "x")
            x_paths = os.listdir(x_dir)
            x_raws = []
            for path in x_paths:
                print(path)
                img_path = os.path.join(x_dir, path)
                x_raw = np.expand_dims(covariance.treat_covariance_with_multiple_window(img_path), axis=2)
                x_raw = np.append(x_raw, np.expand_dims(edge.treat_edge_with_multiple_window(img_path), axis=2), axis=2)
                x_raw = np.append(x_raw, np.expand_dims(smootheness.treat_smoothness(img_path), axis=2), axis=2)
                x_raw = np.append(x_raw, gl.treat_all_glcm_with_multiple_window(img_path), axis=2)
                x_raws.append(x_raw)
                break

            y_dir = os.path.join(data_dir, "y")
            y_paths = os.listdir(y_dir)
            y_raws = []
            for path in y_paths:
                img_path = os.path.join(y_dir, path)
                y_append = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY).astype(np.int32)
                y_append = y_append // 255
                y_raws.append(y_append)
                break

            self.x = np.zeros((0, 9))
            for x_raw in x_raws:
                self.x = np.append(self.x, x_raw.reshape(x_raw.size // 9, 9), axis=0)
            self.x_max = np.max(self.x, axis=0)
            self.x_min = np.min(self.x, axis=0)

            self.y = np.zeros((0, 2))
            for y_raw in y_raws:
                y_raw = y_raw.reshape(y_raw.size)
                y_append = np.zeros((y_raw.size, 2))
                for i in range(y_raw.size):
                    if y_raw[i] > 0:
                        y_append[i] = [0.9, 0.1]
                    else:
                        y_append[i] = [0.1, 0.9]
                self.y = np.append(self.y, y_append, axis=0)

            np.save(os.path.join(data_dir, 'x'), self.x)
            np.save(os.path.join(data_dir, 'x_max'), self.x_max)
            np.save(os.path.join(data_dir, 'x_min'), self.x_min)
            np.save(os.path.join(data_dir, 'y'), self.y)
        else:
            self.x = np.load(os.path.join(data_dir, 'x.npy'))
            self.x_max = np.load(os.path.join(data_dir, 'x_max.npy'))
            self.x_min = np.load(os.path.join(data_dir, 'x_min.npy'))
            self.y = np.load(os.path.join(data_dir, 'y.npy'))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        x = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        x = torch.from_numpy(x)
        y = torch.from_numpy(self.y[idx])
        return x, y


class MLPNetwork(nn.Module):
    def __init__(self, node=16):
        super(MLPNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(9, node),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(node, node),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(node, node),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(node, node),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(node, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.network(x)
        return y


net = MLPNetwork()
x = torch.rand((1, 9))
y = net(x)
print(y.size())
