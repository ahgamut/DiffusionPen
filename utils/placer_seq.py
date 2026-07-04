"""Ordered per-paragraph sequences for the WordPlacer (stage-2 redesign).

Normalization convention
------------------------
All local placement quantities (``gap``, ``base``, ``ink_w``, ``ink_h``) are
expressed in units of ``H`` = the **median word bounding-box height of the
paragraph**. This is a robust, stable scale (unlike the old per-line
``pl_height`` union extent, which is dominated by the single tallest
ascender/descender in the line and varies ~2x line-to-line). Crucially ``H`` is
computed the *same way* at inference -- as the median ink-height of the
generated crops for the paragraph -- so a normalized value means the same thing
in both pipelines. This train/inference agreement is what the earlier
``pl_height`` vs fixed-64 design got wrong.

Line breaking is NOT learned: it is a deterministic greedy fill at inference
(cursor vs ``max_line_width``). The model only predicts, per in-line transition,
the horizontal ``gap`` before a word and its vertical ``base`` drift relative to
its predecessor. Vertical line-advance (new-line height) is a per-writer scalar
statistic (mean line-origin-to-line-origin / ``H``), carried on the model.
"""

import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from utils.subprompt import Word

SEQ_CACHE = "placer_IAM_seq2.pt"
WRITERS_DICT = "utils/writers_dict_train_iam.json"
_PUNCT = set(".,;:!?\"')")


def load_writer_index():
    with open(WRITERS_DICT, "r") as f:
        return json.load(f)


def _after_punct(prev):
    """1.0 if the preceding word ends with punctuation (which affects the
    following inter-word gap), else 0.0."""
    if prev is None or not prev.raw:
        return 0.0
    return 1.0 if prev.raw[-1] in _PUNCT else 0.0


def paragraph_H(doc_words):
    """Robust per-paragraph scale: median word bbox-height. Mirrors the
    inference-time median-crop-height reference. Returns a float >= 1."""
    heights = sorted(max(w.height, 0) for w in doc_words)
    n = len(heights)
    if n == 0:
        return 1.0
    mid = heights[n // 2] if n % 2 else 0.5 * (heights[n // 2 - 1] + heights[n // 2])
    return float(max(mid, 1.0))


def encode_words_text(words, tokenizer, text_encoder, device, max_length=40):
    """Mask-pooled CANINE embedding per word. Returns [N, 768] on ``device``
    (no grad -- the text encoder is frozen)."""
    if len(words) == 0:
        return torch.zeros((0, 768), device=device)
    tok = tokenizer(
        list(words),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        hidden = text_encoder(**tok).last_hidden_state  # [N, L, 768]
    mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [N, L, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled


def sequence_text_features(texts, tokenizer, text_encoder, device, max_length=40):
    """Assemble padded per-sequence text features with a single CANINE pass.

    ``texts`` is a list (len B) of word lists (ragged). Returns
    ``feats`` [B, Tmax, 768] and ``lengths`` (LongTensor [B])."""
    lengths = [len(s) for s in texts]
    tmax = max(lengths) if lengths else 0
    flat = [w for s in texts for w in s]
    pooled = encode_words_text(flat, tokenizer, text_encoder, device, max_length)
    feats = torch.zeros((len(texts), tmax, pooled.shape[-1]), device=device)
    idx = 0
    for b, n in enumerate(lengths):
        feats[b, :n] = pooled[idx : idx + n]
        idx += n
    return feats, torch.tensor(lengths, dtype=torch.long)


def build_sequences(words, writer_index):
    """Group a flat doc-ordered word list into per-paragraph sequences with
    per-word inputs + placement targets (see module docstring for the ``H``
    normalization). Also accumulates dataset-level statistics.

    Returns ``(sequences, stats)`` where ``stats`` holds ``default_gap`` /
    ``default_base`` (mean valid targets, used to center the model's residual
    head) and ``line_advance`` (per-writer-index mean new-line advance in ``H``
    units, plus a global fallback)."""
    docs = OrderedDict()
    for w in words:
        docs.setdefault(w.parent_doc, []).append(w)

    sequences = []
    skipped = 0
    gap_sum = base_sum = trans_n = 0.0
    adv_by_writer = {}  # widx -> [advances]
    for _, doc_words in docs.items():
        wid = doc_words[0].writer_id
        if wid not in writer_index:
            skipped += 1
            continue
        widx = writer_index[wid]
        H = paragraph_H(doc_words)

        seq = []
        prev = None
        # line origin (pl_ystart) per line, in document order, for line-advance
        line_origins = []
        for w in doc_words:
            line_start = 1 if (prev is None or w.parent_line != prev.parent_line) else 0
            if line_start:
                gap = 0.0
                base = 0.0
                line_origins.append(w.pl_ystart)
            else:
                gap = (w.x_start - prev.x_end) / H
                base = (0.5 * (w.y_start + w.y_end) - 0.5 * (prev.y_start + prev.y_end)) / H
                gap_sum += gap
                base_sum += base
                trans_n += 1
            seq.append(
                {
                    "text": w.raw,
                    "writer": widx,
                    "ink_w": w.width / H,
                    "ink_h": w.height / H,
                    "line_start": line_start,
                    "after_punct": _after_punct(prev),
                    "gap": gap,
                    "base": base,
                }
            )
            prev = w

        # per-writer new-line advance: consecutive line-origin gaps / H
        for a, b in zip(line_origins, line_origins[1:]):
            adv_by_writer.setdefault(widx, []).append((b - a) / H)

        if len(seq) >= 2:
            sequences.append(seq)

    if skipped:
        print(f"placer_seq: skipped {skipped} docs with unknown writer")

    default_gap = gap_sum / trans_n if trans_n else 1.0
    default_base = base_sum / trans_n if trans_n else 0.0
    all_adv = [a for v in adv_by_writer.values() for a in v]
    global_adv = sum(all_adv) / len(all_adv) if all_adv else 4.0
    line_advance = {
        widx: (sum(v) / len(v) if v else global_adv) for widx, v in adv_by_writer.items()
    }
    stats = {
        "default_gap": float(default_gap),
        "default_base": float(default_base),
        "line_advance": line_advance,
        "line_advance_global": float(global_adv),
    }
    print(
        "placer_seq: default_gap=%.3f default_base=%.3f line_advance(global)=%.3f"
        % (stats["default_gap"], stats["default_base"], stats["line_advance_global"])
    )
    return sequences, stats


class IAMSequenceDataset(Dataset):
    """One item == one paragraph sequence. Reuses the cached word list in
    ``placer_IAM.pt`` (produced by IAMPlacerDataset) and caches the assembled
    sequences + dataset stats in ``placer_IAM_seq2.pt``."""

    def __init__(self, savefolder="./saved_iam_data"):
        self.savefolder = savefolder
        self.sequences = []
        self.stats = {}
        self.finalize()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

    def finalize(self):
        seq_file = os.path.join(self.savefolder, SEQ_CACHE)
        if os.path.isfile(seq_file):
            blob = torch.load(seq_file, weights_only=False)
            self.sequences = blob["sequences"]
            self.stats = blob["stats"]
            print("loaded save file", seq_file)
        else:
            self.save_sequences(seq_file)
        print(f"dataset has {len(self.sequences)} paragraph sequences")

    def save_sequences(self, seq_file):
        from utils.placer_iam import load_placer_store

        store = load_placer_store(self.savefolder)
        writer_index = load_writer_index()
        self.sequences, self.stats = build_sequences(store.words, writer_index)
        torch.save({"sequences": self.sequences, "stats": self.stats}, seq_file)
        print("saved", seq_file)

    def collate_fn(self, batch):
        lengths = torch.tensor([len(s) for s in batch], dtype=torch.long)
        tmax = int(lengths.max())
        b = len(batch)

        texts = [[w["text"] for w in s] for s in batch]
        writer_ids = torch.tensor([s[0]["writer"] for s in batch], dtype=torch.long)

        ink = torch.zeros((b, tmax, 2), dtype=torch.float32)
        after_punct = torch.zeros((b, tmax), dtype=torch.float32)
        gap = torch.zeros((b, tmax), dtype=torch.float32)
        base = torch.zeros((b, tmax), dtype=torch.float32)
        # trans_mask marks valid in-line transitions: a real word (within length)
        # that is not the first word of its line. These are the only positions
        # with a ground-truth (gap, base); everything else is masked out.
        trans_mask = torch.zeros((b, tmax), dtype=torch.float32)

        for i, s in enumerate(batch):
            for t, w in enumerate(s):
                ink[i, t, 0] = w["ink_w"]
                ink[i, t, 1] = w["ink_h"]
                after_punct[i, t] = w["after_punct"]
                gap[i, t] = w["gap"]
                base[i, t] = w["base"]
                if not w["line_start"]:
                    trans_mask[i, t] = 1.0

        return {
            "texts": texts,
            "writer_ids": writer_ids,
            "ink": ink,
            "after_punct": after_punct,
            "gap": gap,
            "base": base,
            "trans_mask": trans_mask,
            "lengths": lengths,
        }
