# -- coding: utf-8 --
# @Author : Guo

# cascaded newwork

import torch.nn as nn
import torch

myclasses = ["2ASK",
             "4ASK",
             "2FSK",
             "4FSK",
             "BPSK",
             "QPSK",
             "8PSK",
             "16QAM",
             "64QAM",
             "16APSK",
             "32APSK",
             "PAM4",
             "B-FM",
             "DSB-AM",
             "SSB-AM", ]


class ResidualStack(nn.Module):
    def __init__(self, inchannel, maxpoolsize):
        super(ResidualStack, self).__init__()
        self.maxpoolsize = maxpoolsize
        self.conv1 = nn.Conv2d(inchannel, 32, 1, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3, 2), padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 2), padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, (3, 2), padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, (3, 2), padding='same')
        self.bn5 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(maxpoolsize, stride=(2, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # Residual Unit 1
        x_shortcut = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += x_shortcut
        x = self.relu(x)
        # Residual Unit 2
        x_shortcut = x
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x += x_shortcut
        x = self.relu(x)
        # MaxPooling
        x = self.maxpool(x)
        return x


class ConnectedModel_10nofc_2fc(nn.Module):
    def __init__(self, residualstack=ResidualStack, blocks_num=6):
        super(ConnectedModel_10nofc_2fc, self).__init__()
        # ChanNet
        self.conv1 = nn.Conv2d(1, 1, (1, 16), padding=(0, 8))
        self.conv2 = nn.Conv2d(1, 1, (1, 16), padding=(0, 4))
        self.conv3 = nn.Conv2d(1, 1, (1, 16), padding=(0, 2))
        self.conv4 = nn.Conv2d(1, 1, (1, 8), padding=(0, 4))
        self.conv5 = nn.Conv2d(1, 1, (1, 8), padding=(0, 2))
        self.conv6 = nn.Conv2d(1, 1, (1, 8), padding=(0, 0))
        self.conv7 = nn.Conv2d(1, 1, (1, 4), padding=(0, 2))
        self.conv8 = nn.Conv2d(1, 1, (1, 4), padding=(0, 0))
        self.conv9 = nn.Conv2d(1, 1, (1, 4), padding=(0, 0))
        self.conv10 = nn.Conv2d(1, 1, (1, 2), padding=(0, 0))
        # ModNet
        self.blocks_num = blocks_num
        self.residual_layer1 = residualstack(1, (2, 2))
        self.residual_layer2 = residualstack(32, (2, 1))
        self.residual_layer3 = residualstack(32, (2, 1))
        self.residual_layer4 = residualstack(32, (2, 1))
        self.residual_layer5 = residualstack(32, (2, 1))
        self.residual_layer6 = residualstack(32, (2, 1))
        self.flat = nn.Flatten()
        self.line1m = nn.Linear(512, 128)
        self.line2m = nn.Linear(128, len(myclasses))
        self.selu = nn.SELU()
        self.alphadropout = nn.AlphaDropout()
        # self.softmax = nn.Softmax(dim=1)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # print('intermediate size', x.shape)
        x = torch.transpose(x, 3, 2)
        # print('transpose', x.shape)
        x = self.residual_layer1(x)
        x = self.residual_layer2(x)
        x = self.residual_layer3(x)
        x = self.residual_layer4(x)
        x = self.residual_layer5(x)
        x = self.residual_layer6(x)
        x = self.flat(x)
        x = self.line1m(x)
        x = self.selu(x)
        x = self.alphadropout(x)
        x = self.line2m(x)
        return x
