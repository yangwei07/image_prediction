import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Hyper Parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TIME_STEP = 4
BATCH_SIZE = 10


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            # conv1
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # conv2
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_out = nn.Linear(64 * 64 * 64, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_out = nn.Linear(128, 2)
        self.bn1 = torch.nn.BatchNorm1d(64 * 64 * 64)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size = x.shape[0]
        seq = torch.empty(batch_size, TIME_STEP, 64).to(DEVICE)
        for i in range(TIME_STEP):
            s = self.cnn(x[:, i, :, :, :])
            s = s.view(s.size(0), -1)
            s = self.bn1(s)
            seq[:, i, :] = self.cnn_out(s)
        r_out, (h_n, h_c) = self.lstm(seq, None)
        s = self.bn2(r_out[:, -1, :])
        out = self.lstm_out(s)
        return out


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txtfile):
        fh = open(txtfile, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1], words[2]))
        self.imgs = imgs
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor()]
        )

    def __getitem__(self, index):
        fn, x, y = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        img = self.transform(img)
        return img, (x, y)

    def __len__(self):
        return len(self.imgs)


class DataProcess:
    def __init__(self):
        self.num = 0
        self.img = None
        self.pos = None

    def data_prepare(self, user):
        f = open('data.txt', 'w')
        for root, dirs, _ in os.walk("./image"):
            for _, docname in enumerate(dirs):
                for path, _, files in os.walk(os.path.join(root, docname)):
                    labels = pd.read_csv(os.path.join(path, 'data.csv'), header=None).iloc[user * 20:(user + 1) * 20].values
                    i = 0
                    for _, filename in enumerate(files):
                        if filename[-3:] == 'jpg':
                            filepath = os.path.join(path, 'im_' + str(i + 1) + '.jpg')
                            f.write(filepath + ' ' + str(labels[i, 0]) + ' ' + str(labels[i, 1]) + '\n')
                            i += 1
        f.close()

    def load_tensor(self):
        train_data = MyDataset('data.txt')
        loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)
        self.num = len(loader)
        self.img = torch.empty(1, self.num, 3, 256, 256)
        self.pos = torch.empty(self.num, 2)
        for i, data in enumerate(loader):
            inputs, labels = data
            labels = torch.FloatTensor(np.array(labels).astype(float).transpose())
            self.img[:, i, :, :, :] = inputs
            self.pos[i, :] = labels[-1, :]

    def sample_tensor(self, train):
        if train:
            samples = np.random.choice(self.num - TIME_STEP, BATCH_SIZE)
        else:
            samples = range(4, 20)
        n = len(samples)
        inputs_batch = torch.empty(n, TIME_STEP, 3, 256, 256)
        labels_batch = torch.empty(n, 2)
        for i, j in enumerate(samples):
            index = list(range(j, j + TIME_STEP))
            inputs_batch[i, :, :, :, :] = self.img[:, index, :, :, :]
            labels_batch[i, :] = self.pos[index[-1], :]
        return inputs_batch, labels_batch


if __name__ == '__main__':
    d = DataProcess()
    results_loss = []
    results_error = []
    results_pred = []
    results_grth = []
    for user in range(25):
        d.data_prepare(user)  # uncomment this line on OSX
        d.load_tensor()
        rnn = CNN_LSTM().to(DEVICE)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
        loss_func = nn.MSELoss()
        # train neural network
        running_loss = []
        for epoch in range(200):
            inputs_batch, labels_batch = d.sample_tensor(True)
            optimizer.zero_grad()
            outputs_batch = rnn(inputs_batch.to(DEVICE))
            loss = loss_func(outputs_batch, labels_batch.to(DEVICE))
            loss.backward()
            optimizer.step()
            running_loss.append(loss.data.to('cpu').numpy() / BATCH_SIZE)
            print('epoch: %d| loss: %.3f' % (epoch + 1, loss / BATCH_SIZE))
        results_loss.append(running_loss)
        # test neural network
        inputs_batch, labels_batch = d.sample_tensor(False)
        outputs_batch = rnn(inputs_batch.to(DEVICE))
        prediction = outputs_batch.data.to('cpu').numpy()
        groundtruth = labels_batch.data.numpy()
        num_test = prediction.shape[0]
        error = 0.
        for i in range(num_test):
            error += np.sqrt((prediction[i, 0] - groundtruth[i, 0]) ** 2 + (prediction[i, 1] - groundtruth[i, 1]) ** 2)
        results_error.append(error)
        results_pred.append([prediction[:, 0], prediction[:, 1]])
        results_grth.append([groundtruth[:, 0], groundtruth[:, 1]])
    data = pd.DataFrame(results_loss)
    data.to_csv('loss.csv')
    data = pd.DataFrame(results_error)
    data.to_csv('error.csv')
    data = pd.DataFrame(results_pred)
    data.to_csv('pred.csv')
    data = pd.DataFrame(results_grth)
    data.to_csv('grth.csv')

