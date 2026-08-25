import torch
import torch.nn as nn


class WordPlacer(nn.Module):
    """Per-transition word placement (stage-2 redesign).

    Line breaking is *not* learned (a deterministic greedy fill handles it at
    inference). For each adjacent in-line pair ``(prev, cur)`` the model predicts
    a small Gaussian over two placement residuals, in units of the paragraph
    scale ``H`` (median word height; see utils/placer_seq.py):

      - ``gap``  horizontal whitespace before ``cur`` (cur.x_start - prev.x_end)
      - ``base`` vertical baseline/center drift of ``cur`` relative to ``prev``

    Outputs are ``mu``/``logvar`` per quantity; ``mu`` is a residual added to a
    dataset default so the net learns deviations, not absolute scale. Sampling
    ``~ N(mu, exp(logvar))`` (done by the caller) gives the required slight
    randomness. There is no RNN: the prediction for position ``t`` depends only
    on words ``t-1`` (prev) and ``t`` (cur), which keeps it simple and avoids the
    baseline-drift accumulation the old GRU tried (and failed) to model.

    ``forward`` inputs (position ``t`` is the transition prev=t-1 -> cur=t):
      - ``text_feats``   [B, T, text_dim]  frozen CANINE pooled embedding per word
      - ``style_vec``    [B, style_dim] or [B, T, style_dim]  frozen per-writer
                                            style-bank vector (same space the
                                            diffusion UNet consumes)
      - ``ink_dims``     [B, T, 2]          normalized (ink_w, ink_h) per word
      - ``after_punct``  [B, T]             1 if prev word ended with punctuation
      - ``lengths``      unused (kept for call-site compatibility)

    Outputs (each [B, T]): ``mu_gap, logvar_gap, mu_base, logvar_base``. Position
    0 and any line-start position have no valid predecessor; the caller masks
    them (training) or overrides them with the line-advance (inference).

    The per-writer new-line advance is not predicted here but carried as the
    ``line_advance`` buffer (in ``H`` units) so it travels with the checkpoint;
    ``default_gap`` / ``default_base`` buffers hold the residual centers. All
    three are populated from dataset statistics by the training script.
    """

    def __init__(
        self,
        num_writers=339,
        style_dim=1280,
        text_dim=768,
        text_reduced=64,
        writer_dim=32,
        hidden=128,
        default_gap=1.2,
        default_base=0.0,
        default_advance=4.0,
        logvar_min=-6.0,
        logvar_max=2.0,
    ):
        super().__init__()
        self.text_reduce = nn.Linear(text_dim, text_reduced)
        # Writer conditioning is a projection of the frozen style-bank vector
        # (style-only redesign), NOT a per-writer embedding: learns a shared
        # style->placement map that also serves writers with a bank row but no
        # paragraph data (CVL/CSAFE), and regularizes vs. the old lookup table.
        self.style_proj = nn.Sequential(nn.Linear(style_dim, writer_dim), nn.ReLU())
        # input = writer_feat + reduced(prev) + reduced(cur) + ink(prev,2)
        #         + ink(cur,2) + after_punct(1)
        input_dim = writer_dim + 2 * text_reduced + 2 + 2 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 4)  # d_mu_gap, logvar_gap, d_mu_base, logvar_base
        self.act = nn.ReLU()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # Statistics carried with the checkpoint (not optimized).
        self.register_buffer("default_gap", torch.tensor(float(default_gap)))
        self.register_buffer("default_base", torch.tensor(float(default_base)))
        self.register_buffer(
            "line_advance", torch.full((num_writers,), float(default_advance))
        )

    def _shift_prev(self, x):
        """Return a tensor whose position t holds x[t-1] (position 0 -> zeros)."""
        prev = torch.zeros_like(x)
        prev[:, 1:] = x[:, :-1]
        return prev

    def forward(self, text_feats, style_vec, ink_dims, after_punct=None, lengths=None):
        b, t, _ = text_feats.shape
        if after_punct is None:
            after_punct = torch.zeros((b, t), device=text_feats.device)

        tred_cur = self.act(self.text_reduce(text_feats))
        tred_prev = self._shift_prev(tred_cur)
        ink_prev = self._shift_prev(ink_dims)

        wemb = self.style_proj(style_vec)
        if wemb.dim() == 2:  # one style vector per sequence -> broadcast over time
            wemb = wemb.unsqueeze(1).expand(-1, t, -1)

        x = torch.cat(
            [wemb, tred_prev, tred_cur, ink_prev, ink_dims, after_punct.unsqueeze(-1)],
            dim=-1,
        )
        out = self.head(self.net(x))

        mu_gap = self.default_gap + out[..., 0]
        logvar_gap = out[..., 1].clamp(self.logvar_min, self.logvar_max)
        mu_base = self.default_base + out[..., 2]
        logvar_base = out[..., 3].clamp(self.logvar_min, self.logvar_max)
        return mu_gap, logvar_gap, mu_base, logvar_base
