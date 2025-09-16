import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from inspect import isfunction
import math


class HorizontalPlacer(nn.Module):
    # if the images of consecutive words are provided,
    # find out how the second word must be placed after the first.
    # return offsets as coefficients (c_w, c_h), where
    #
    # if TOP_LEFT(cur_image) is at (0, 0),
    #    TOP_LEFT(next_image) should be at (c_w * HEIGHT, c_h * HEIGHT)
    def __init__(self, text_encoder, style_encoder, hidden_size=768, style_size=1280):
        super().__init__()
        self.text_encoder = text_encoder
        self.style_encoder = style_encoder

        # combine text/style features
        self.combi = nn.Bilinear(hidden_size, style_size, 640)
        # return spacing coefficients
        self.rspce = nn.Bilinear(640, 640, 2)
        self.ac = nn.Tanh()

    def forward(self, x_cur, x_next):
        tch_1 = self.text_encoder(**x_cur["text_features"]).last_hidden_state
        tch_1 = tch_1.mean(dim=1)
        sty_1 = self.style_encoder(x_cur["image"])
        # print(tch_1.shape, sty_1.shape)
        x_1 = self.combi(tch_1, sty_1)
        x_1 = self.ac(x_1)

        tch_2 = self.text_encoder(**x_next["text_features"]).last_hidden_state
        sty_2 = self.style_encoder(x_next["image"])
        tch_2 = tch_2.mean(dim=1)
        # print(tch_2.shape, sty_2.shape)
        x_2 = self.combi(tch_2, sty_2)
        x_2 = self.ac(x_2)
        # print(x_1.shape, x_2.shape)

        coeffs = self.rspce(x_1, x_2)
        return coeffs
