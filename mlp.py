import os
import time

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


class SeismicAttributeDataset(tdata.Dataset):
    def __init__(self, data_usage, is_new):
        data_dir = os.path.join("mlp_dataset", data_usage)

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

            y_dir = os.path.join(data_dir, "y")
            y_paths = os.listdir(y_dir)
            y_raws = []
            for path in y_paths:
                img_path = os.path.join(y_dir, path)
                y_append = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY).astype(np.int32)
                y_append = y_append // 255
                y_raws.append(y_append)

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
            self.x_max = np.load(os.path.join("mlp_dataset", 'x_max.npy'))
            self.x_min = np.load(os.path.join("mlp_dataset", 'x_min.npy'))
            self.y = np.load(os.path.join(data_dir, 'y.npy'))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        x = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        x = torch.from_numpy(x)
        y = torch.from_numpy(self.y[idx])
        return x, y


def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0.0)


class MLPNetwork(nn.Module):
    def __init__(self, node=16, drop=0.5):
        super(MLPNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(9, node),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(drop),
            nn.Linear(node, node),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(drop),
            nn.Linear(node, node),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(drop),
            nn.Linear(node, node),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(drop),
            nn.Linear(node, 2),
            nn.Softmax(dim=1)
        )

        self.apply(init_weights)

    def forward(self, x):
        y = self.network(x)
        return y


def construct_dataset():
    SeismicAttributeDataset(data_usage="train", is_new=True)
    SeismicAttributeDataset(data_usage="valid", is_new=True)
    SeismicAttributeDataset(data_usage="test", is_new=True)

    train_x_max = np.load(os.path.join("mlp_dataset", "train", 'x_max.npy'))
    train_x_min = np.load(os.path.join("mlp_dataset", "train", 'x_min.npy'))

    valid_x_max = np.load(os.path.join("mlp_dataset", "valid", 'x_max.npy'))
    valid_x_min = np.load(os.path.join("mlp_dataset", "valid", 'x_min.npy'))

    test_x_max = np.load(os.path.join("mlp_dataset", "test", 'x_max.npy'))
    test_x_min = np.load(os.path.join("mlp_dataset", "test", 'x_min.npy'))

    x_max = np.append(train_x_max.reshape((9, 1)), valid_x_max.reshape((9, 1)), axis=1)
    x_max = np.append(x_max, test_x_max.reshape((9, 1)), axis=1)

    x_min = np.append(train_x_min.reshape((9, 1)), valid_x_min.reshape((9, 1)), axis=1)
    x_min = np.append(x_min, test_x_min.reshape((9, 1)), axis=1)

    x_max = np.max(x_max, axis=1)
    x_min = np.min(x_min, axis=1)

    np.save(os.path.join("mlp_dataset", 'x_max'), x_max)
    np.save(os.path.join("mlp_dataset", 'x_min'), x_min)


def train():
    cudnn.benchmark = True
    tag = "mlp"
    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda_available else "cpu")

    n_epoch = 100
    s_epoch = 0
    lr = 1e-3
    batch_size = 256
    gamma = 0.995

    node = 16
    drop = 0.5

    network = MLPNetwork(node, drop=drop).to(device).double()

    train_set = SeismicAttributeDataset(data_usage="train", is_new=False)
    train_loader = tdata.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    valid_set = SeismicAttributeDataset(data_usage="valid", is_new=False)
    valid_loader = tdata.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    criterion = nn.KLDivLoss(reduction="batchmean")

    train_losses = []
    valid_losses = []
    accuracies = []
    if s_epoch != 0:
        state = torch.load(os.path.join("mlp_dataset", "network", tag + "_" + str(s_epoch).zfill(4)))

        network.load_state_dict(state["network"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        train_losses = state["train_losses"]
        valid_losses = state["valid_losses"]
        accuracies = state["accuracies"]

    if len(train_set) % batch_size == 0:
        n_train = len(train_loader)
    else:
        n_train = len(train_loader) - 1

    if len(valid_set) % batch_size == 0:
        n_valid = len(valid_loader)
    else:
        n_valid = len(valid_loader) - 1

    ave_train_loss = 0.0
    ave_valid_loss = 0.0
    ema_coeff = 0.9
    for i_epoch in range(s_epoch, n_epoch):
        start_time = time.time()

        network.train()
        ave_train_loss_biased = 0.0
        for i_batch, (x_data, y_data) in enumerate(train_loader):
            if x_data.size()[0] != batch_size:
                continue

            optimizer.zero_grad()

            x = x_data.to(device)
            label = y_data.to(device)
            y = network(x)
            loss = criterion(torch.log(y), label)
            loss.backward()
            optimizer.step()

            ave_train_loss_biased = ema_coeff * ave_train_loss_biased + (1 - ema_coeff) * loss.item()
            if (i_batch + 1) % 100 == 0 or (i_batch + 1) == n_train:
                ave_train_loss = ave_train_loss_biased / (1 - ema_coeff ** (i_batch + 1))
                print("epoch: {:4}, batch index: {:4}, train loss: {:7.4f}".format(
                    i_epoch + 1,
                    i_batch + 1,
                    ave_train_loss
                ))

        network.eval()
        crr_cnt = 0
        ttl_cnt = 0
        ave_valid_loss_biased = 0.0
        accuracy = 0
        for i_batch, (x_data, y_data) in enumerate(valid_loader):
            if x_data.size()[0] != batch_size:
                continue

            x = x_data.to(device)
            label = y_data.to(device)
            y = network(x)
            loss = criterion(torch.log(y), label)

            _, prediction = torch.max(y.data, dim=1)
            _, answer = torch.max(label.data, dim=1)
            ttl_cnt += x.data.size()[0]
            for i_data in range(x.data.size()[0]):
                if prediction[i_data] == answer[i_data]:
                    crr_cnt += 1

            ave_valid_loss_biased = ema_coeff * ave_valid_loss_biased + (1 - ema_coeff) * loss.item()
            if (i_batch + 1) % 100 == 0 or (i_batch + 1) == n_valid:
                ave_valid_loss = ave_valid_loss_biased / (1 - ema_coeff ** (i_batch + 1))
                accuracy = float(crr_cnt) / ttl_cnt
                print("epoch: {:4}, batch index: {:4}, valid loss: {:7.4f}, accuracy: {:.3f}".format(
                    i_epoch + 1,
                    i_batch + 1,
                    ave_valid_loss,
                    accuracy
                ))

        scheduler.step()
        train_losses.append(ave_train_loss)
        valid_losses.append(ave_valid_loss)
        accuracies.append(accuracy)

        state = {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "accuracies": accuracies
        }

        print("epoch: {:4}, save training state".format(i_epoch + 1))
        torch.save(state, os.path.join("mlp_dataset", "network", tag + "_" + str(i_epoch + 1).zfill(4)))

        print("epoch: {:4}, execution time: {:6.2f}".format(i_epoch + 1, time.time() - start_time))


if __name__ == "__main__":
    train()
