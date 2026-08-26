"""Siamese writer discriminator head (evaluation statistics).

Given two per-set style vectors (each the mean of a writer's K crop features from
the frozen ``ImageEncoder``), decide same-writer vs different-writer. The head is
the classic siamese form: a single linear layer over the absolute elementwise
difference ``|v1 - v2|``, trained with ``BCEWithLogitsLoss`` (same=1, different=0).

Kept intentionally tiny: the discriminative power lives in the frozen style
extractor; this only calibrates a decision boundary on top of it.
"""

import torch.nn as nn


class WriterDiscriminator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, v1, v2):
        # v1, v2: [B, feat] (each already averaged across a writer's K crops).
        # Returns raw logits [B]; use BCEWithLogitsLoss for training, sigmoid for
        # a same-writer probability at inference.
        return self.fc((v1 - v2).abs()).squeeze(-1)
