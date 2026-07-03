import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np

from models import WordUpsampler, AvgMeter
from utils.generation import setup_logging
from utils.arghandle import add_common_args


class IAMCropDataset(Dataset):
    """Real IAM word crops (the cached ``wimgs`` from placer_IAM.pt) used as HR
    targets for self-supervised super-resolution."""

    def __init__(self, savefolder="./saved_iam_data", size=(64, 256)):
        self.size = size
        base_file = os.path.join(savefolder, "placer_IAM.pt")
        raw = torch.load(base_file, weights_only=False)
        self.wimgs = raw["wimgs"]

    def __len__(self):
        return len(self.wimgs)

    def __getitem__(self, index):
        h, w = self.size
        img = Image.frombytes(mode="RGB", size=(w, h), data=self.wimgs[index])
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]


def make_lr(hr, scale):
    """Bicubic-downsample HR -> LR (the self-supervision input)."""
    return F.interpolate(
        hr, scale_factor=1.0 / scale, mode="bicubic", align_corners=False
    )


def run_batch(hr, model, scale, loss_fn, args):
    hr = hr.to(args.device)
    lr = make_lr(hr, scale)
    sr = model(lr)
    return loss_fn(sr, hr)


def main():
    parser = argparse.ArgumentParser("upsampler-train")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    if args.dataset != "iam":
        raise ValueError("upsampler_train only supports the IAM dataset")

    dset = IAMCropDataset()
    train_size = int(0.8 * len(dset))
    test_size = len(dset) - train_size
    train_data, test_data = random_split(
        dset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.dataparallel:
        device_ids = [3, 4]
    else:
        idx = int("".join(filter(str.isdigit, args.device)))
        device_ids = [idx]

    model = WordUpsampler(scale=args.scale)
    model = DataParallel(model, device_ids=device_ids)
    model = model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    ckpt_path = os.path.join(args.save_path, "models", "upsampler_ckpt.pt")
    optim_path = os.path.join(args.save_path, "models", "upsampler_optim.pt")
    if args.load_check and os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        if os.path.isfile(optim_path):
            optimizer.load_state_dict(torch.load(optim_path, weights_only=True))

    train_meter = AvgMeter("L1")
    val_meter = AvgMeter("L1")
    print("Training started....")
    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        model.train()
        train_meter.reset()
        for hr in train_loader:
            loss = run_batch(hr, model, args.scale, loss_fn, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(loss.item(), hr.size(0))
        print("train", repr(train_meter))

        if epoch % 10 == 0:
            model.eval()
            val_meter.reset()
            with torch.no_grad():
                for hr in test_loader:
                    loss = run_batch(hr, model, args.scale, loss_fn, args)
                    val_meter.update(loss.item(), hr.size(0))
            print("validation", repr(val_meter))
            torch.save(model.state_dict(), ckpt_path)
            torch.save(optimizer.state_dict(), optim_path)


if __name__ == "__main__":
    main()
