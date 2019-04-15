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
TEST_SIZE = 16
DATA_READY = False


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            # conv1
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_out = nn.Linear(64 * 64 * 64, 64)
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


def dataset(category):
    img = []
    seq = []
    pos = []
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(256), transforms.ToTensor()])
    for root, dirs, _ in os.walk(os.path.join('./image', str(category))):
        for _, docname in enumerate(dirs):
            for path, _, files in os.walk(os.path.join(root, docname)):
                labels = []
                for _, f in enumerate(files):
                    if f[-3:] == 'csv':
                        labels.extend(pd.read_csv(os.path.join(path, f), header=None).values)
                labels = np.array(labels)
                a = labels.any(1) != 0
                labels = labels[labels.any(1) != 0]
                n = int(labels.shape[0] / 20)
                for i in range(n):
                    for j in range(20 - TIME_STEP + 1):
                        img_path = os.path.join(path, 'im_' + str(j + TIME_STEP) + '.jpg')
                        image = Image.open(img_path).convert('RGB')
                        img.append(transform(image))
                        seq.append(torch.FloatTensor(labels[i * 20 + j:i * 20 + j + TIME_STEP - 1, :]))
                        pos.append(torch.FloatTensor(labels[i * 20 + j + TIME_STEP - 1, :]))
    num = len(img)
    with open('data' + str(category) + '.pkl', 'wb') as data_output:
        pickle.dump((img, seq, pos, num), data_output)


def train(data, rnn, category):
    img = torch.empty(BATCH_SIZE, 3, 256, 256).to(DEVICE)
    seq = torch.empty(BATCH_SIZE, 3, 2).to(DEVICE)
    pos = torch.empty(BATCH_SIZE, 2).to(DEVICE)
    num = data[3]
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-5)
    loss_func = nn.MSELoss()
    running_loss = []
    for episode in range(5000):
        samples = np.random.choice(range(TEST_SIZE, num), BATCH_SIZE)
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
        print('category: %d | episode: %d| loss: %.3f' % (category, episode + 1, loss / BATCH_SIZE))
    torch.save(rnn.state_dict(), 'params' + str(category) + '.pkl')
    file = pd.DataFrame(running_loss)
    file.to_csv('loss' + str(category) + '.csv')
    print('training finishes!')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(running_loss)
    # ax.set_xlabel('episode')
    # ax.set_ylabel('loss')


def eval(data, rnn, category):
    rnn.load_state_dict(torch.load('params' + str(category) + '.pkl'))
    img = torch.empty(TEST_SIZE, 3, 256, 256).to(DEVICE)
    seq = torch.empty(TEST_SIZE, 3, 2).to(DEVICE)
    pos = torch.empty(TEST_SIZE, 2).to(DEVICE)
    samples = range(TEST_SIZE)
    for i, index in enumerate(samples):
        img[i, :, :, :] = data[0][index]
        seq[i, :, :] = data[1][index]
        pos[i, :] = data[2][index]
    output = rnn(img, seq)
    prediction = output.data.to('cpu').numpy()
    groundtruth = pos.data.to('cpu').numpy()
    test_data = np.concatenate((prediction, groundtruth), 1)
    file = pd.DataFrame(test_data)
    file.to_csv('test' + str(category) + '.csv')
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(range(TEST_SIZE), prediction[:, 0], prediction[:, 1], label='prediction')
    # ax.plot(range(TEST_SIZE), groundtruth[:, 0], groundtruth[:, 1], label='groundtruth')
    # ax.legend()
    # ax.set_xlabel('samples')
    # ax.set_ylabel('x')
    # ax.set_zlabel('y')


if __name__ == '__main__':
    # for category in range(1):
    category = 2
    if DATA_READY is False:
        dataset(category)
    with open('data' + str(category) + '.pkl', 'rb') as file:
        data = pickle.load(file)
    rnn = CNN_LSTM().to(DEVICE)
    train(data, rnn, category)
    eval(data, rnn, category)
