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
    def __init__(self, text_encoder, style_encoder, hidden_size=768, style_size=256):
        self.text_encoder = text_encoder
        self.style_encoder = style_encoder

        # combine text/style features
        self.combi = nn.Bilinear(hidden_size, style_size, 1280)
        # return spacing coefficients
        self.rspce = nn.Bilinear(1280, 1280, 2)
        self.ac = nn.Tanh()

    def forward(self, x_cur, x_next):
        tch_1 = self.text_encoder(**x_cur["text_feature"]).last_hidden_state
        sty_1 = self.style_encoder(x_cur["image"])
        x_1 = self.combi(tch_1, sty_1)
        x_1 = self.ac(x_1)

        tch_2 = self.text_encoder(**x_next["text_feature"]).last_hidden_state
        sty_2 = self.style_encoder(x_next["image"])
        x_2 = self.combi(tch_2, sty_2)
        x_2 = self.ac(x_2)

        coeffs = self.rspce(x_1, x_2)
        return coeffs
