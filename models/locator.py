import torch
import torch.nn as nn


class WordLocator(nn.Module):
    # given the text of a word,
    # find out its size and how it should be located,
    # in terms of the width/height of the line it is on
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        # reduce dimensions
        self.conv1 = nn.Conv1d(
            in_channels=40, out_channels=48, kernel_size=16, stride=8
        )
        self.conv2 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=16, stride=8
        )
        # return spacing coefficients
        self.rspc0 = nn.Linear(128, 32)
        self.rspc1 = nn.Linear(32, 3)
        self.ac = nn.Sigmoid()
        self.ac2 = nn.ReLU()

    def forward(self, word_features):
        tch_1 = self.text_encoder(**word_features).last_hidden_state  # N x 40 x 768
        tch_1 = self.ac2(self.conv1(tch_1))  # N x 48 x 95
        tch_1 = self.ac2(self.conv2(tch_1))  # N x 64 x 22
        tch_1 = self.ac2(self.conv3(tch_1))  # N x 128 x 1
        tch_1 = tch_1.reshape(tch_1.shape[0], -1)  # N x 128
        x1 = self.ac2(self.rspc0(tch_1))  # N x 32
        x1 = self.ac(self.rspc1(x1))  # N x 3
        return x1
