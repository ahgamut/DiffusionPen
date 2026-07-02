import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import sys
import os
import argparse
import random
import scipy.stats as scs
import pandas as pd

#
from utils.placer_iam import RelWordIndices
from utils.subprompt import Word
from utils.arghandle import add_common_args

#

ALL_CAPS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
STICK_UP = set("bdfhklt'\"!?")  # expect these to increase height upwards
STICK_DN = set("fgjpqy,.;")  # expect these to increase height downwards
PUNCT_ST = set(",./;:'\"[]!@#$%^&*()-_+=\\|")
NUMBERS = set("0123456789")


class WordLocationDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        raw = torch.load(filename, weights_only=False)
        self.words = [Word.from_bytes(x) for x in raw["words"]]
        self.max_wordlen = 40

        wids = set(x.writer_id for x in self.words)
        self.windex_forward = dict()
        self.windex_backward = dict()
        for i, w in enumerate(sorted(wids)):
            self.windex_forward[w] = i
            self.windex_backward[i] = w

    def __len__(self):
        return len(self.words)

    def make_targets(self, word):
        nwidth = word.nwidth
        nheight = word.nheight
        relystart = (word.y_start - word.pl_ystart) / word.pl_height
        vec = {"nwidth": nwidth, "nheight": nheight, "rystart": relystart}
        return vec

    def __getitem__(self, index):
        cur_word = self.words[index]
        targs = self.make_targets(cur_word)
        wid0 = self.windex_forward[cur_word.writer_id]
        wids = torch.tensor([wid0], dtype=torch.int64)

        return wids, cur_word.raw, torch.tensor([1]), targs


class SimplePlacerDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        raw = torch.load(filename, weights_only=False)
        self.words = [Word.from_bytes(x) for x in raw["words"]]
        self.pairs = [RelWordIndices.from_bytes(x) for x in raw["pairs"]]
        self.max_wordlen = 40

        wids = set(x.writer_id for x in self.words)
        self.windex_forward = dict()
        self.windex_backward = dict()
        for i, w in enumerate(sorted(wids)):
            self.windex_forward[w] = i
            self.windex_backward[i] = w
        # print(wids)

    def __len__(self):
        return len(self.pairs)

    def make_covariates(self, cur_word, next_word):
        # print(cur_word, next_word)
        cur_rawset = set(cur_word.raw)
        cur_len = len(cur_word.raw) / 40
        next_rawset = set(next_word.raw)
        next_len = len(next_word.raw) / 40
        vec = [
            cur_len,
            stick(cur_rawset, ALL_CAPS),
            stick(cur_rawset, STICK_UP),
            stick(cur_rawset, STICK_DN),
            stick(cur_rawset, PUNCT_ST),
            stick(cur_rawset, NUMBERS),
            next_len,
            stick(next_rawset, ALL_CAPS),
            stick(next_rawset, STICK_UP),
            stick(next_rawset, STICK_DN),
            stick(next_rawset, PUNCT_ST),
            stick(next_rawset, NUMBERS),
        ]
        return torch.tensor(vec)

    def make_targets(self, cur_word, next_word):
        space_x = (next_word.x_start - cur_word.x_end) / cur_word.pl_width
        hdiff_y = (next_word.y_start - cur_word.y_start) / cur_word.pl_height
        next_height = next_word.nheight
        next_ystart = (next_word.y_start - next_word.pl_ystart) / next_word.pl_height
        vec = [space_x, hdiff_y, next_height, next_ystart]
        return vec

    def __getitem__(self, index):
        rwi = self.pairs[index]
        cur_word = self.words[rwi.cur_index]
        next_word = self.words[rwi.next_index]

        targs = self.make_targets(cur_word, next_word)
        wid0 = self.windex_forward[cur_word.writer_id]
        wids = torch.tensor([wid0], dtype=torch.int64)

        return wid0, (cur_word.raw, next_word.raw), targs

    def collate_fn(self, batch):
        wids, covs, targs = zip(*batch)
        wids = torch.cat(wids)
        covs = torch.stack(covs)
        targs = torch.cat(targs)
        return wids, covs, targs


def eladd(d, w, x):
    if w not in d.keys():
        d[w] = [x]
    else:
        d[w].append(x)


def xspace_distro(dset):
    res = []
    for wid, cnw, cov, targ in dset:
        pqc = set([x for x in cnw])
        if len(pqc - PUNCT_ST) == 0:
            if len(cnw) > 1:
                continue
            targ["word"] = cnw
            res.append(targ)

    rdf = pd.DataFrame(res)
    r0 = rdf[rdf["nwidth"] <= 0.05]
    for g, gdf in r0.groupby("word"):
        if len(gdf) < 10:
            print(g, "skip")
            continue
        print(g, gdf.describe())


def main():
    parser = argparse.ArgumentParser("simple-placer")
    add_common_args(parser)

    args = parser.parse_args()
    if args.dataset == "iam":
        dset = WordLocationDataset("./saved_iam_data/placer_IAM_wpo.pt")
    else:
        raise RuntimeError(f"{args.dataset}: can't load dataset!")

    xspace_distro(dset)


if __name__ == "__main__":
    main()
