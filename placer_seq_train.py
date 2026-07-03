import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

from models import WordPlacer, AvgMeter
from utils.placer_seq import IAMSequenceDataset, sequence_text_features
from utils.generation import setup_logging
from utils.arghandle import add_common_args
from utils.training_utils import custom_loss, get_loaders


def frz(model):
    model.eval()
    model.requires_grad_(False)


def run_batch(batch, placer, tokenizer, text_encoder, args):
    device = args.device
    text_feats, lengths = sequence_text_features(
        batch["texts"], tokenizer, text_encoder, device
    )
    writer_ids = batch["writer_ids"].to(device)
    ink = batch["ink"].to(device)
    nl_logit, x_gap, y_off = placer(text_feats, writer_ids, ink, lengths=lengths)

    mask = batch["mask"].to(device)
    tgt_nl = batch["newline"].to(device)
    tgt_xg = batch["x_gap"].to(device)
    tgt_yo = batch["y_off"].to(device)

    bce = nn.functional.binary_cross_entropy_with_logits(
        nl_logit, tgt_nl, reduction="none"
    )
    loss_nl = (bce * mask).sum() / mask.sum().clamp(min=1.0)

    # zero out padded positions so they contribute nothing to the regression loss
    reg_loss_fn = custom_loss(0.01, alpha=1.0, beta=5.0)
    pred_reg = torch.stack([x_gap * mask, y_off * mask], dim=-1)
    tgt_reg = torch.stack([tgt_xg * mask, tgt_yo * mask], dim=-1)
    loss_reg = reg_loss_fn(pred_reg, tgt_reg) / mask.sum().clamp(min=1.0)

    return loss_nl + loss_reg


def train_epoch(placer, tokenizer, text_encoder, optimizer, loader, meter, args):
    placer.train()
    meter.reset()
    for batch in loader:
        loss = run_batch(batch, placer, tokenizer, text_encoder, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), int(batch["lengths"].numel()))
    print("train", repr(meter))


def val_epoch(placer, tokenizer, text_encoder, loader, meter, args):
    placer.eval()
    meter.reset()
    with torch.no_grad():
        for batch in loader:
            loss = run_batch(batch, placer, tokenizer, text_encoder, args)
            meter.update(loss.item(), int(batch["lengths"].numel()))
    print("validation", repr(meter))


def main():
    parser = argparse.ArgumentParser("placer-seq-train")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    if args.dataset != "iam":
        raise ValueError("placer_seq_train only supports the IAM dataset")

    dset = IAMSequenceDataset()
    train_loader, test_loader = get_loaders(dset, args.batch_size)

    if args.dataparallel:
        device_ids = [3, 4]
    else:
        idx = int("".join(filter(str.isdigit, args.device)))
        device_ids = [idx]

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = DataParallel(text_encoder, device_ids=device_ids)
    text_encoder = text_encoder.to(args.device)
    frz(text_encoder)

    placer = WordPlacer(num_writers=339)
    placer = DataParallel(placer, device_ids=device_ids)
    placer = placer.to(args.device)

    optimizer = optim.AdamW(placer.parameters(), lr=args.lr)

    ckpt_path = os.path.join(args.save_path, "models", "placer_seq_ckpt.pt")
    optim_path = os.path.join(args.save_path, "models", "placer_seq_optim.pt")
    if args.load_check and os.path.isfile(ckpt_path):
        placer.load_state_dict(torch.load(ckpt_path, weights_only=True))
        if os.path.isfile(optim_path):
            optimizer.load_state_dict(torch.load(optim_path, weights_only=True))

    train_meter = AvgMeter("loss")
    val_meter = AvgMeter("loss")
    print("Training started....")
    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        train_epoch(
            placer, tokenizer, text_encoder, optimizer, train_loader, train_meter, args
        )
        if epoch % 10 == 0:
            val_epoch(placer, tokenizer, text_encoder, test_loader, val_meter, args)
            torch.save(placer.state_dict(), ckpt_path)
            torch.save(optimizer.state_dict(), optim_path)


if __name__ == "__main__":
    main()
