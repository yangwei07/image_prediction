import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


# Hyper Parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TIME_STEP = 4
BATCH_SIZE = 128
TEST_SIZE = 20
TEST_PERCENT = .8
DATA_READY = True


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # conv2
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_out = nn.Linear(128 * 64 * 64, 64)
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(128, 2)
        self.bn = torch.nn.BatchNorm1d(128)

    def forward(self, img, pos):
        batch_size = img.shape[0]
        s1 = self.cnn(img)  # [batch, C, H, W]
        s1 = s1.view(batch_size, -1)
        c_out = self.cnn_out(s1)
        seq = torch.empty(batch_size, TIME_STEP - 1, 32).to(DEVICE)
        for i in range(TIME_STEP - 1):
            s2 = self.fc1(pos[:, i, :].squeeze())  # [batch, time, P]
            s2 = self.relu(s2)
            s2 = self.fc2(s2)
            seq[:, i, :] = s2
        r_out, (_, _) = self.lstm(seq, None)
        s = torch.cat((r_out[:, -1, :], c_out), 1)
        s = self.bn(s)
        out = self.out(s)
        return out


def dataset():
    img = []
    seq = []
    pos = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor()]
    )
    for root, dirs, _ in os.walk("./image"):
        for _, docname in enumerate(dirs):
            for path, _, files in os.walk(os.path.join(root, docname)):
                for _, f in enumerate(files):
                    if f[-3:] == 'csv':
                        labels = pd.read_csv(os.path.join(path, f), header=None)
                        n = int(labels.shape[0] / 20)
                for i in range(n):
                    for j in range(17):
                        img_path = os.path.join(path, 'im_' + str(j + TIME_STEP) + '.jpg')
                        image = Image.open(img_path).convert('RGB')
                        img.append(transform(image))
                        seq.append(torch.FloatTensor(labels.iloc[i * 20 + j:i * 20 + j + TIME_STEP - 1].values))
                        pos.append(torch.FloatTensor(labels.iloc[i * 20 + j + TIME_STEP - 1].values))
    num = len(img)
    with open('data.pkl', 'wb') as data_output:
        pickle.dump((img, seq, pos, num), data_output)


def train(data, rnn):
    img = torch.empty(BATCH_SIZE, 3, 256, 256).to(DEVICE)
    seq = torch.empty(BATCH_SIZE, 3, 2).to(DEVICE)
    pos = torch.empty(BATCH_SIZE, 2).to(DEVICE)
    num = data[3]
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-5)
    loss_func = nn.MSELoss()
    running_loss = []
    for episode in range(5000):
        samples = np.random.choice(range(0, int(num * TEST_PERCENT)), BATCH_SIZE)
        for i, index in enumerate(samples):
            img[i, :, :, :] = data[0][index]
            seq[i, :, :] = data[1][index]
            pos[i, :] = data[2][index]
        optimizer.zero_grad()
        output = rnn(img, seq)
        loss = loss_func(output, pos)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.data.to('cpu').numpy() / BATCH_SIZE)
        print('episode: %d| loss: %.3f' % (episode + 1, loss / BATCH_SIZE))
    torch.save(rnn.state_dict(), 'params.pkl')
    data = pd.DataFrame(running_loss)
    data.to_csv('loss.csv')
    print('training finishes!')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(running_loss)
    ax.set_xlabel('episode')
    ax.set_ylabel('loss')


def eval(data, rnn):
    rnn.load_state_dict(torch.load('params.pkl'))
    img = torch.empty(TEST_SIZE, 3, 256, 256).to(DEVICE)
    seq = torch.empty(TEST_SIZE, 3, 2).to(DEVICE)
    pos = torch.empty(TEST_SIZE, 2).to(DEVICE)
    num = data[3]
    samples = np.random.choice(range(int(num * TEST_PERCENT), num), TEST_SIZE)
    for i, index in enumerate(samples):
        img[i, :, :, :] = data[0][index]
        seq[i, :, :] = data[1][index]
        pos[i, :] = data[2][index]
    output = rnn(img, seq)
    prediction = output.data.to('cpu').numpy()
    groundtruth = pos.data.to('cpu').numpy()
    test_data = np.concatenate((prediction, groundtruth), 1)
    data = pd.DataFrame(test_data)
    data.to_csv('test.csv')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(range(TEST_SIZE), prediction[:, 0], prediction[:, 1], label='prediction')
    ax.plot(range(TEST_SIZE), groundtruth[:, 0], groundtruth[:, 1], label='groundtruth')
    ax.legend()
    ax.set_xlabel('samples')
    ax.set_ylabel('x')
    ax.set_zlabel('y')


if __name__ == '__main__':
    if DATA_READY is False:
        dataset()
    with open('data.pkl', 'rb') as file:
        data = pickle.load(file)
    rnn = CNN_LSTM().to(DEVICE)
    train(data, rnn)
    # eval(data, rnn)
