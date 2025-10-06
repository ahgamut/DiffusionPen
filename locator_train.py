import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import sys
import os
import argparse
from transformers import CanineModel, CanineTokenizer

#
from models import WordLocator
from utils.placer_iam import RelWordIndices
from utils.subprompt import Word
from utils.arghandle import add_common_args

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

        return wids, cur_word.raw, targs

    def collate_fn(self, batch):
        wids, covs, targs = zip(*batch)
        wids = torch.cat(wids)
        targs = torch.cat(targs)
        return wids, covs, targs


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


def train_epoch(
    model, tokenizer, text_encoder, loader, loss_fn, optimizer, loss_meter, args
):
    model.train()
    for i, data in enumerate(loader):
        wids = data[0]  # .to(args.device)
        words = data[1]
        text_features = tokenizer(
            words,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        ).to(args.device)
        targs = data[2].to(args.device)
        preds = model(text_features)

        loss = loss_fn(targs, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_meter.update(loss, targs.shape[0])
    print("train", repr(loss_meter))


def val_epoch(
    model, tokenizer, text_encoder, loader, loss_fn, optimizer, loss_meter, args
):
    model.eval()
    big_err = 0
    for i, data in enumerate(loader):
        wids = data[0]  # .to(args.device)
        words = data[1]
        text_features = tokenizer(
            words,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        ).to(args.device)
        targs = data[2].to(args.device)
        preds = model(text_features)

        loss = loss_fn(targs, preds)
        errs = torch.abs(preds - targs)
        big_err = max(big_err, torch.quantile(errs[:, 0], 0.9).item())
        # print(preds - targs)
        # print(loss)
        loss_meter.update(loss, targs.shape[0])
    print("val", repr(loss_meter), big_err)


def train(
    model,
    tokenizer,
    text_encoder,
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
        train_epoch(
            model,
            tokenizer,
            text_encoder,
            train_loader,
            loss_fn,
            optimizer,
            loss_meter,
            args,
        )

        if i % 10 == 0:
            val_epoch(
                model,
                tokenizer,
                text_encoder,
                test_loader,
                loss_fn,
                optimizer,
                loss_meter,
                args,
            )
            torch.save(
                model.state_dict(),
                os.path.join(wts_dir, "models", "locator_ckpt.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(wts_dir, "models", "locator_optim.pt"),
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
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
    text_encoder = text_encoder.to(args.device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    model = WordLocator(text_encoder)
    model = nn.DataParallel(model, device_ids=device_ids)
    model_wts_path = f"{args.save_path}/models/locator_ckpt.pt"
    if os.path.isfile(model_wts_path):
        model.load_state_dict(torch.load(model_wts_path, weights_only=True))
    model = model.to(args.device)

    ####
    loss_fn = custom_loss(0.01, alpha=0.5, beta=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_meter = AvgMeter("MSE")

    ####
    train(
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
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
