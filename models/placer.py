import torch.nn as nn


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
        self.seq_encoder = nn.RNN(
            input_size=hidden_size,
            hidden_size=64,
            batch_first=True,
            bidirectional=False,
            num_layers=5,
        )

        # combine text/style features
        self.combi = nn.Bilinear(64, style_size, 128)
        # return spacing coefficients
        self.rspce = nn.Bilinear(128, 128, 1)
        self.ac = nn.Tanh()
        self.ac2 = nn.Softshrink(lambd=1.0)

    def forward(self, x_cur, x_next):
        tch_1 = self.text_encoder(**x_cur["text_features"]).last_hidden_state
        _, nx_1 = self.seq_encoder(tch_1)
        nx_1 = nx_1.mean(dim=0)
        sty_1 = self.style_encoder(x_cur["image"])
        # print(tch_1.shape, nx_1.shape, sty_1.shape)
        x_1 = self.combi(nx_1, sty_1)
        x_1 = self.ac(x_1)

        tch_2 = self.text_encoder(**x_next["text_features"]).last_hidden_state
        sty_2 = self.style_encoder(x_next["image"])
        _, nx_2 = self.seq_encoder(tch_2)
        # print(tch_2.shape, sty_2.shape)
        nx_2 = nx_2.mean(dim=0)
        x_2 = self.combi(nx_2, sty_2)
        x_2 = self.ac(x_2)
        # print(x_1.shape, x_2.shape)

        coeffs = self.rspce(x_1, x_2)
        coeffs = self.ac2(coeffs)
        return coeffs
