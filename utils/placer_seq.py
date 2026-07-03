"""Ordered per-paragraph sequences for the autoregressive WordPlacer.

Normalization convention
------------------------
All local placement quantities (``x_gap``, ``y_off``, ``ink_w``, ``ink_h``) are
expressed in units of the **line height** of the word's own line — ``pl_height``
in IAM, which is a per-line quantity. This is the one scale that is consistent
between training (IAM bboxes at native resolution) and inference (generated
crops whose ink height is ~one rendered line). At inference the line-height
reference is a fixed ``REF_HEIGHT`` (64 px), so a normalized value is turned back
into pixels by multiplying by 64.

Note ``Word.pl_width`` is overwritten by ``Prompt.__init__`` to the *paragraph*
width; we deliberately do NOT use it here. Page width only bounds line wrapping
at inference (``max_line_width``); it is not a local-placement basis.
"""

import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from utils.subprompt import Word

REF_HEIGHT = 64
WRITERS_DICT = "utils/writers_dict_train_iam.json"


def load_writer_index():
    with open(WRITERS_DICT, "r") as f:
        return json.load(f)


def encode_words_text(words, tokenizer, text_encoder, device, max_length=40):
    """Mask-pooled CANINE embedding per word. Returns [N, 768] on ``device``
    (no grad — the text encoder is frozen)."""
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
    per-word inputs + placement targets. Words whose IAM writer id is not in the
    (train) writer index are skipped whole-document."""
    docs = OrderedDict()
    for w in words:
        docs.setdefault(w.parent_doc, []).append(w)

    sequences = []
    skipped = 0
    for _, doc_words in docs.items():
        wid = doc_words[0].writer_id
        if wid not in writer_index:
            skipped += 1
            continue
        widx = writer_index[wid]

        seq = []
        prev = None
        for w in doc_words:
            plh = max(w.pl_height, 1)
            newline = 1 if (prev is None or w.parent_line != prev.parent_line) else 0
            if newline:
                x_gap = 0.0
                if prev is None:
                    y_off = 0.0
                else:
                    # inter-line advance (line origins), in prev line-heights
                    y_off = (w.pl_ystart - prev.pl_ystart) / max(prev.pl_height, 1)
            else:
                x_gap = (w.x_start - prev.x_end) / plh
                y_off = (w.y_start - prev.y_start) / plh
            seq.append(
                {
                    "text": w.raw,
                    "writer": widx,
                    "ink_w": w.width / plh,
                    "ink_h": w.height / plh,
                    "newline": newline,
                    "x_gap": x_gap,
                    "y_off": y_off,
                }
            )
            prev = w
        if len(seq) >= 2:
            sequences.append(seq)
    if skipped:
        print(f"placer_seq: skipped {skipped} docs with unknown writer")
    return sequences


class IAMSequenceDataset(Dataset):
    """One item == one paragraph sequence. Reuses the cached word list in
    ``placer_IAM.pt`` (produced by IAMPlacerDataset) and caches the assembled
    sequences in ``placer_IAM_seq.pt``."""

    def __init__(self, savefolder="./saved_iam_data"):
        self.savefolder = savefolder
        self.sequences = []
        self.finalize()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

    def finalize(self):
        seq_file = os.path.join(self.savefolder, "placer_IAM_seq.pt")
        if os.path.isfile(seq_file):
            self.sequences = torch.load(seq_file, weights_only=False)
            print("loaded save file", seq_file)
        else:
            self.sequences = self.save_sequences(seq_file)
        print(f"dataset has {len(self.sequences)} paragraph sequences")

    def save_sequences(self, seq_file):
        base_file = os.path.join(self.savefolder, "placer_IAM.pt")
        raw = torch.load(base_file, weights_only=False)
        words = [Word.from_bytes(x) for x in raw["words"]]
        writer_index = load_writer_index()
        sequences = build_sequences(words, writer_index)
        torch.save(sequences, seq_file)
        print("saved", seq_file)
        return sequences

    def collate_fn(self, batch):
        lengths = torch.tensor([len(s) for s in batch], dtype=torch.long)
        tmax = int(lengths.max())
        b = len(batch)

        texts = [[w["text"] for w in s] for s in batch]
        writer_ids = torch.tensor([s[0]["writer"] for s in batch], dtype=torch.long)

        ink = torch.zeros((b, tmax, 2), dtype=torch.float32)
        newline = torch.zeros((b, tmax), dtype=torch.float32)
        x_gap = torch.zeros((b, tmax), dtype=torch.float32)
        y_off = torch.zeros((b, tmax), dtype=torch.float32)
        mask = torch.zeros((b, tmax), dtype=torch.float32)

        for i, s in enumerate(batch):
            for t, w in enumerate(s):
                ink[i, t, 0] = w["ink_w"]
                ink[i, t, 1] = w["ink_h"]
                newline[i, t] = w["newline"]
                x_gap[i, t] = w["x_gap"]
                y_off[i, t] = w["y_off"]
                mask[i, t] = 1.0

        return {
            "texts": texts,
            "writer_ids": writer_ids,
            "ink": ink,
            "newline": newline,
            "x_gap": x_gap,
            "y_off": y_off,
            "mask": mask,
            "lengths": lengths,
        }
