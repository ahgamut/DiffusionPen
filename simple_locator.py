import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import sys
import os
import argparse
from transformers import CanineModel, CanineTokenizer

#
from utils.placer_iam import RelWordIndices
from utils.subprompt import Word
from utils.arghandle import add_common_args
from utils.auxilary_functions import get_default_character_classes

#

ALL_CAPS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
STICK_UP = set("bdfhklt'\"!?")  # expect these to increase height upwards
STICK_DN = set("fgjpqy,.;")  # expect these to increase height downwards
PUNCT_ST = set(",./;:'\"[]!@#$%^&*()-_+=\\|")
NUMBERS = set("0123456789")


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def stick(w, ws):
    return float(len(w & ws) > 0) - 0.5


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

        self.charmap = {x: i for (i, x) in enumerate(self.char_classes)}
        self.char_classes = get_default_character_classes()
        self.dummies = [len(self.charmap) - 1] * 40

    def __len__(self):
        return len(self.words)

    def make_targets(self, word):
        nwidth = word.nwidth
        nheight = word.nheight
        relystart = (word.y_start - word.pl_ystart) / word.pl_height
        vec = [nwidth, nheight, relystart]
        return torch.tensor([vec])

    def __getitem__(self, index):
        cur_word = self.words[index]
        targs = self.make_targets(cur_word)
        wid0 = self.windex_forward[cur_word.writer_id]
        wids = torch.tensor([wid0], dtype=torch.int64)

        idx = [self.charmap[x] for x in cur_word.raw] + self.dummies[
            len(cur_word.raw) :
        ]

        return wids, cur_word.raw, torch.tensor(idx), targs

    def collate_fn(self, batch):
        wids, words, covs, targs = zip(*batch)
        wids = torch.cat(wids)
        covs = torch.stack(covs)
        targs = torch.cat(targs)
        return wids, words, covs, targs


class SimpleLocator(nn.Module):
    def __init__(self, char_features, max_wordlen=40):
        self.max_wordlen = max_wordlen
        self.char_classes = get_default_character_classes()
        self.char_features = char_features
        self.embedding = nn.Embedding(
            len(self.char_classes),
            char_features,
            padding_idx=len(self.char_classes) - 1,
        )
        self.l1 = nn.Linear(max_wordlen * char_features, 16)
        self.l2 = nn.Linear(16, 3)
        self.ac = nn.ReLU()

    def forward(self, idxs):
        xf = self.embedding(idxs)
        xf = xf.reshape(xf.shape[0], -1)
        xf = self.ac(self.l1(xf))
        xf = self.ac(self.l2(xf))
        return xf


def get_loaders(dset, batch_size):
    train_size = int(0.8 * len(dset))
    test_size = len(dset) - train_size
    train_data, test_data = random_split(dset, [train_size, test_size])
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=dset.collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        shuffle=True,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=dset.collate_fn,
    )
    return train_loader, test_loader


def train_epoch(model, loader, loss_fn, optimizer, loss_meter, args):
    model.train()
    for i, data in enumerate(loader):
        wids = data[0]  # .to(args.device)
        words = data[1]
        idxs = data[2].to(args.device)
        targs = data[3].to(args.device)
        preds = model(idxs)

        loss = loss_fn(targs, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_meter.update(loss, targs.shape[0])
    print("train", repr(loss_meter))


def val_epoch(model, loader, loss_fn, optimizer, loss_meter, args):
    model.eval()
    big_err = 0
    for i, data in enumerate(loader):
        wids = data[0]  # .to(args.device)
        words = data[1]
        idxs = data[2].to(args.device)
        targs = data[3].to(args.device)
        preds = model(idxs)

        loss = loss_fn(targs, preds)
        errs = torch.abs(preds - targs)
        big_err = max(big_err, torch.quantile(errs[:, 0], 0.9).item())
        # print(preds - targs)
        # print(loss)
        loss_meter.update(loss, targs.shape[0])
    print("val", repr(loss_meter), big_err)


def train(
    model,
    epochs,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    loss_meter,
    wts_dir,
    args,
):
    for i in range(epochs + 1):
        print("epoch:", i, end=" ")
        train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            loss_meter,
            args,
        )

        if i % 10 == 0:
            val_epoch(
                model,
                test_loader,
                loss_fn,
                optimizer,
                loss_meter,
                args,
            )
            torch.save(
                model.state_dict(),
                os.path.join(wts_dir, "models", "simloc_ckpt.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(wts_dir, "models", "simloc_optim.pt"),
            )


def custom_loss(out=1.0, alpha=0.5, beta=2.0):
    def fn(pred, target):
        l2 = nn.functional.mse_loss(pred, target, reduction="none")
        small = alpha * l2[l2 <= out].sum()
        big = beta * l2[l2 > out].sum()
        return big + small

    return fn


def main():
    parser = argparse.ArgumentParser("locator-train")
    parser.add_argument("-b", "--batch-size", type=int, default=10, help="size")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="epochs")
    parser.add_argument("--char-features", default=8, type=int, help="char features")
    add_common_args(parser)

    args = parser.parse_args()
    if args.dataset == "iam":
        dset = WordLocationDataset("./saved_iam_data/placer_IAM_wpo.pt")
    else:
        raise RuntimeError(f"{args.dataset}: can't load dataset!")
    train_loader, test_loader = get_loaders(dset, args.batch_size)

    if args.dataparallel == True:
        device_ids = [3, 4]
        print("using dataparallel with device:", device_ids)
    else:
        idx = int("".join(filter(str.isdigit, args.device)))
        device_ids = [idx]

    ####
    model = SimpleLocator(args.char_features)
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(args.device)

    ####
    loss_fn = custom_loss(0.01, alpha=1.0, beta=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_meter = AvgMeter("MSE")

    ####
    train(
        model=model,
        epochs=args.epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loss_meter=loss_meter,
        wts_dir=args.save_path,
        args=args,
    )


if __name__ == "__main__":
    main()
