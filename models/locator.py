import torch.nn as nn


class WordLocator(nn.Module):
    # given the text of a word,
    # find out its size and how it should be located,
    # in terms of the width/height of the line it is on
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        # return spacing coefficients
        self.rspc0 = nn.Linear(768, 32)
        self.rspc1 = nn.Linear(32, 3)
        self.ac = nn.Sigmoid()

    def forward(self, word_features):
        tch_1 = self.text_encoder(**word_features).last_hidden_state
        tch_1 = tch_1.mean(dim=1)
        # print(tch_1.shape) # N x 768
        x1 = self.rspc0(tch_1)
        x1 = self.ac(x1)
        x1 = self.rspc1(x1)
        x1 = self.ac(x1)

        return x1
