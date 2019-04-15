import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 256, 256)
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # output shape (32, 128, 128)
        )
        self.conv2 = nn.Sequential(  # input shape (32, 128, 128)
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 64, 64)
        )

        self.out = nn.Linear(64 * 64 * 64, 2)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # (batch_size, 64 * 64 * 64)
        output = self.out(x)
        return output


if __name__ == '__main__':
    pos = pd.read_csv('dev_0_v01_BabyPandas.csv', header=None)
    trainset = torchvision.datasets.ImageFolder('train_data',
                                                transform=transforms.Compose([
                                                    transforms.Resize((256, 256)),
                                                    transforms.CenterCrop(256),
                                                    transforms.ToTensor()])
                                                )
    loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    cnn = CNN().to(DEVICE)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    loss_func = nn.MSELoss()
    # train neural network
    running_loss = []
    for epoch in range(5):
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            labels = torch.FloatTensor(pos.ix[labels].values)
            # wrap them in Variable
            inputs = Variable(inputs).to(DEVICE)
            labels = Variable(labels).to(DEVICE)
            optimizer.zero_grad()
            outputs = cnn(inputs).to(DEVICE)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = np.append(running_loss, loss.data.to('cpu'))
            print('epoch: %d| batch: %d| loss: %.3f'
                  % (epoch + 1, i + 1, loss / BATCH_SIZE,))

    print('Finished Training')
    plt.figure()
    plt.plot(running_loss)
    plt.title('running loss')
    plt.show()
    # test neural network

    testset = torchvision.datasets.ImageFolder('test_data',
                                               transform=transforms.Compose([
                                                   transforms.Resize((256, 256)),
                                                   transforms.CenterCrop(256),
                                                   transforms.ToTensor()])
                                               )
    loader = torch.utils.data.DataLoader(testset)
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        labels = pos.ix[labels].values[0]
        inputs = Variable(inputs).to(DEVICE)
        outputs = cnn(inputs).to(DEVICE)
        outputs = outputs.data.to('cpu').numpy()[0]
        print('prediction: [%.3f %.3f] | groundtruth: [%.3f %.3f]' %
              (outputs[0], outputs[1], labels[0], labels[1]))


