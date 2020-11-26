import os

import numpy as np
import torch
import torch.utils.data as tdata
from PIL import Image
import cv2

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
            np.save(os.path.join(data_dir, 'y'), self.y)
        else:
            self.x = np.load(os.path.join(data_dir, 'x.npy'))
            self.y = np.load(os.path.join(data_dir, 'y.npy'))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


a = SeismicAttributeDataset(is_test=True, is_new=False)
print(a.__getitem__(1))
