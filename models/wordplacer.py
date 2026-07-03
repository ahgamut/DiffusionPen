import torch
import torch.nn as nn


class WordPlacer(nn.Module):
    """Autoregressive next-word placement over a paragraph.

    The model walks a paragraph left-to-right, carrying a running cursor/baseline
    context in the RNN hidden state, and for each word predicts *placement
    decisions only*: whether the word starts a new line, the horizontal gap
    before it, and a vertical offset. Word ink width/height are **inputs** (taken
    from the already-generated crop), not outputs.

    Inputs to ``forward``:
      - ``text_feats``  [B, T, text_dim]  frozen CANINE pooled embedding per word
      - ``writer_ids``  [B] or [B, T]     IAM style-class index (0..num_writers-1)
      - ``ink_dims``    [B, T, 2]          normalized (ink_w, ink_h) per word
      - ``lengths``     optional [B]       true sequence lengths (for packing)

    Outputs (each [B, T]):
      - ``newline_logit`` (apply sigmoid for p_newline)
      - ``x_gap``         normalized horizontal gap before the word
      - ``y_off``         normalized vertical offset (baseline drift / line advance)
    """

    def __init__(
        self,
        num_writers=339,
        text_dim=768,
        text_reduced=64,
        writer_dim=32,
        hidden=128,
        num_layers=2,
    ):
        super().__init__()
        self.text_reduce = nn.Linear(text_dim, text_reduced)
        self.writer_emb = nn.Embedding(num_writers, writer_dim)
        input_dim = text_reduced + writer_dim + 2
        self.input_proj = nn.Linear(input_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 3)
        self.act = nn.ReLU()

    def forward(self, text_feats, writer_ids, ink_dims, lengths=None):
        b, t, _ = text_feats.shape
        tred = self.act(self.text_reduce(text_feats))

        wemb = self.writer_emb(writer_ids)
        if wemb.dim() == 2:  # one writer per sequence -> broadcast over time
            wemb = wemb.unsqueeze(1).expand(-1, t, -1)

        x = torch.cat([tred, wemb, ink_dims], dim=-1)
        x = self.act(self.input_proj(x))

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=t
            )
        else:
            out, _ = self.rnn(x)

        out = self.head(out)
        return out[..., 0], out[..., 1], out[..., 2]
