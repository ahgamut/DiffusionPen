import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

from models import WordPlacer, AvgMeter
from utils.placer_seq import (
    DEFAULT_WRITERS_DIR,
    IAMSequenceDataset,
    load_num_writers,
    sequence_text_features,
)
from utils.generation import setup_logging
from utils.arghandle import add_common_args
from utils.training_utils import get_loaders


def frz(model):
    model.eval()
    model.requires_grad_(False)


def gaussian_nll(target, mu, logvar):
    """Per-element negative log-likelihood of ``target`` under N(mu, exp(logvar)),
    dropping the constant 0.5*log(2*pi). Lower logvar => sharper, but the
    ``(target-mu)^2 * exp(-logvar)`` term penalizes overconfidence."""
    return 0.5 * (logvar + (target - mu) ** 2 * torch.exp(-logvar))


def run_batch(batch, placer, tokenizer, text_encoder, style_bank, args):
    device = args.device
    text_feats, lengths = sequence_text_features(
        batch["texts"], tokenizer, text_encoder, device
    )
    writer_ids = batch["writer_ids"].to(device)
    # Writer conditioning = the frozen style-bank vector, looked up the same way
    # inference does (style-only placer). style_bank is [W, style_dim] on device.
    style_vec = style_bank[writer_ids]
    ink = batch["ink"].to(device)
    after_punct = batch["after_punct"].to(device)
    mu_gap, logvar_gap, mu_base, logvar_base = placer(
        text_feats, style_vec, ink, after_punct=after_punct, lengths=lengths
    )

    mask = batch["trans_mask"].to(device)
    tgt_gap = batch["gap"].to(device)
    tgt_base = batch["base"].to(device)

    # Gaussian NLL on gap + baseline residuals, over valid in-line transitions.
    nll = gaussian_nll(tgt_gap, mu_gap, logvar_gap) + gaussian_nll(
        tgt_base, mu_base, logvar_base
    )
    loss = (nll * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def populate_stat_buffers(placer, stats):
    """Copy dataset statistics into the model's (non-optimized) buffers so they
    ship in the checkpoint: residual centers + per-writer new-line advance."""
    core = placer.module if isinstance(placer, DataParallel) else placer
    with torch.no_grad():
        core.default_gap.fill_(stats["default_gap"])
        core.default_base.fill_(stats["default_base"])
        core.line_advance.fill_(stats["line_advance_global"])
        for widx, val in stats["line_advance"].items():
            core.line_advance[int(widx)] = val
    print("populated placer stat buffers from dataset")


def train_epoch(placer, tokenizer, text_encoder, optimizer, loader, meter, style_bank, args):
    placer.train()
    meter.reset()
    for batch in loader:
        loss = run_batch(batch, placer, tokenizer, text_encoder, style_bank, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), int(batch["lengths"].numel()))
    print("train", repr(meter))


def val_epoch(placer, tokenizer, text_encoder, loader, meter, style_bank, args):
    placer.eval()
    meter.reset()
    with torch.no_grad():
        for batch in loader:
            loss = run_batch(batch, placer, tokenizer, text_encoder, style_bank, args)
            meter.update(loss.item(), int(batch["lengths"].numel()))
    print("validation", repr(meter))


def main():
    parser = argparse.ArgumentParser("placer-seq-train")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_WRITERS_DIR,
        help="merged split dir holding writers_global.json (global-W id space + "
        "writer count for the placer's embedding)",
    )
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    if args.dataset != "iam":
        raise ValueError("placer_seq_train only supports the IAM dataset")

    # Global writer count W (from the merged split); the placer shares this id
    # space with the style bank so a single --writer-id is consistent at
    # inference. Only IAM writers get trained rows (paragraph data is IAM-only).
    num_writers = load_num_writers(args.data_dir)
    dset = IAMSequenceDataset(writers_dir=args.data_dir)
    train_loader, test_loader = get_loaders(dset, args.batch_size)

    # Writer conditioning is the frozen style bank (style-only placer): the same
    # [W, style_dim] tensor inference loads, so a single --writer-id means the
    # same style to both. Required here (no trained per-writer embedding anymore).
    bank_path = args.style_bank_path
    if not os.path.isfile(bank_path):
        raise FileNotFoundError(
            "placer training needs the style bank at {} -- build it with "
            "utils/build_style_bank.py against --data-dir (same --style-name as the "
            "deployed model).".format(bank_path)
        )
    style_bank = torch.load(bank_path, map_location=args.device, weights_only=True)
    if style_bank.shape[0] != num_writers:
        raise ValueError(
            "style bank writer count {} != W {}; the bank must be built from the "
            "same merged split as writers_global.json".format(
                style_bank.shape[0], num_writers
            )
        )
    style_bank = style_bank.to(args.device)
    style_dim = int(style_bank.shape[1])
    print("loaded style bank", bank_path, tuple(style_bank.shape))

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

    placer = WordPlacer(num_writers=num_writers, style_dim=style_dim)
    placer = DataParallel(placer, device_ids=device_ids)
    placer = placer.to(args.device)
    # Seed the stat buffers from the dataset before any (optional) checkpoint
    # load, so a fresh run saves them and a resumed run keeps the ckpt's values.
    populate_stat_buffers(placer, dset.stats)

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
            placer, tokenizer, text_encoder, optimizer, train_loader, train_meter,
            style_bank, args,
        )
        if epoch % 10 == 0:
            val_epoch(
                placer, tokenizer, text_encoder, test_loader, val_meter, style_bank, args
            )
            torch.save(placer.state_dict(), ckpt_path)
            torch.save(optimizer.state_dict(), optim_path)


if __name__ == "__main__":
    main()
