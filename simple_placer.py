import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import sys
import os
import argparse

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
        space_x = next_word.x_start - cur_word.x_end
        hdiff_y = next_word.y_start - cur_word.y_start
        cur_height = cur_word.height
        vec = [hdiff_y / 50]
        return torch.tensor(vec)

    def __getitem__(self, index):
        rwi = self.pairs[index]
        cur_word = self.words[rwi.cur_index]
        next_word = self.words[rwi.next_index]

        covs = self.make_covariates(cur_word, next_word)
        targs = self.make_targets(cur_word, next_word)
        wids = torch.tensor(
            [self.windex_forward[cur_word.writer_id]], dtype=torch.int64
        )

        return wids, covs, targs

    def collate_fn(self, batch):
        wids, covs, targs = zip(*batch)
        wids = torch.cat(wids)
        covs = torch.stack(covs)
        targs = torch.cat(targs)
        return wids, covs, targs


class SimplePlacer(nn.Module):
    def __init__(self, in_features, out_features, total_wids):
        super().__init__()
        self.wid_weights = nn.Embedding(total_wids, 16)

        self.ac = nn.ReLU()

        self.l1 = nn.Linear(in_features, 24)
        self.l2 = nn.Bilinear(24, 16, 32)
        self.l3 = nn.Linear(32, out_features)

    def forward(self, x):
        wids, covs = x
        w = self.wid_weights(wids)
        w = w.reshape(w.shape[0], -1)

        z = self.l1(covs)
        z = self.ac(z)
        z = self.l2(z, w)
        z = self.ac(z)
        z = self.l3(z)
        return z


def get_loaders(dset, batch_size):
    train_size = int(0.8 * len(dset))
    test_size = len(dset) - train_size
    train_data, test_data = random_split(dset, [train_size, test_size])
    train_loader = DataLoader(
        train_data, shuffle=True, num_workers=2, batch_size=batch_size
    )
    test_loader = DataLoader(
        test_data, shuffle=True, num_workers=2, batch_size=batch_size
    )
    return train_loader, test_loader


def train_epoch(model, loader, loss_fn, optimizer, loss_meter):
    model.train()
    for i, data in enumerate(loader):
        widcov = data[:2]
        targs = data[2]
        preds = model(widcov)

        loss = loss_fn(targs, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_meter.update(loss, targs.shape[0])
    print("train", repr(loss_meter))


def val_epoch(model, loader, loss_fn, optimizer, loss_meter):
    model.eval()
    big_err = 0
    for i, data in enumerate(loader):
        widcov = data[:2]
        targs = data[2]
        preds = model(widcov)

        loss = loss_fn(targs, preds)
        errs = torch.abs(preds - targs)
        big_err = max(big_err, torch.quantile(errs[:, 0], 0.9).item())
        # print(preds - targs)
        # print(loss)
        loss_meter.update(loss, targs.shape[0])
    print("val", repr(loss_meter), big_err)


def train(
    model, epochs, train_loader, test_loader, loss_fn, optimizer, loss_meter, wts_dir
):
    for i in range(epochs + 1):
        train_epoch(model, train_loader, loss_fn, optimizer, loss_meter)

        if i % 10 == 0:
            val_epoch(model, test_loader, loss_fn, optimizer, loss_meter)
            torch.save(
                model.state_dict(),
                os.path.join(wts_dir, "models", "simplace_ckpt.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(wts_dir, "models", "simplace_optim.pt"),
            )


def custom_loss(out=1.0, alpha=0.5, beta=2.0):
    def fn(pred, target):
        l2 = nn.functional.mse_loss(pred, target, reduction="none")
        small = alpha * l2[l2 <= out].sum()
        big = beta * l2[l2 > out].sum()
        return big + small

    return fn


def main():
    parser = argparse.ArgumentParser("simple-placer")
    parser.add_argument("-b", "--batch-size", type=int, default=10, help="size")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="epochs")
    add_common_args(parser)

    args = parser.parse_args()
    if args.dataset == "iam":
        dset = SimplePlacerDataset(args.input_data)
    else:
        raise RuntimeError(f"{args.dataset}: can't load dataset!")
    train_loader, test_loader = get_loaders(dset, args.batch_size)

    if args.dataparallel == True:
        device_ids = [3, 4]
        print("using dataparallel with device:", device_ids)
    else:
        idx = int("".join(filter(str.isdigit, args.device)))
        device_ids = [idx]

    model = SimplePlacer(
        in_features=12, out_features=1, total_wids=len(dset.windex_forward)
    )
    model = DataParallel(model, device_ids=device_ids)
    model_wts_path = f"{args.save_path}/models/simplace_ckpt.pt"
    if os.path.isfile(model_wts_path):
        model.load_state_dict(torch.load(model_wts_path, weights_only=True))
    loss_fn = custom_loss(0.02, alpha=0.5, beta=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_meter = AvgMeter("MSE")

    model = model.to(args.device)

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
