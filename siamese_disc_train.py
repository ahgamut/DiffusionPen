"""Train the siamese writer discriminator (evaluation statistics).

On top of the frozen style extractor (the same ``ImageEncoder`` used by the style
bank), train a tiny head to tell same-writer from different-writer, given two
sets of K word crops. Each set is encoded and mean-pooled into one style vector;
the head scores ``|v1 - v2|`` with ``BCEWithLogitsLoss`` (same=1, different=0).

The extractor is frozen, so all crop features are precomputed once into an
``[N, feat]`` cache and the sampling loop only gathers/averages rows -- no repeated
CNN passes. Run on a GPU box after building a merged split:

    python siamese_disc_train.py --data-dir saved_iam_data/combined_word_train \\
        --style-name resnet18 --style-path <style.pth> --device cuda:0 \\
        --out ./style_models/writer_discriminator.pth
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from models import WriterDiscriminator, AvgMeter
from utils.word_dataset import MergedWordDataset, _sample
from utils.build_style_bank import load_extractor, build_feature_cache
from utils.arghandle import add_common_args


def _set_vector(feats, pool, k):
    """Mean feature over a K-crop sample of a writer's pool -> [feat]."""
    return feats[_sample(pool, k)].mean(dim=0)


def build_pair_batch(feats, dataset, writer_ids, args):
    """One training batch: for each of ``batch_size`` anchors emit a same-writer
    positive pair (label 1) and a different-writer negative pair (label 0)."""
    v1, v2, labels = [], [], []
    for _ in range(args.batch_size):
        a = random.choice(writer_ids)
        pool_a = dataset._same_writer_pool(a)
        # positive: two independent K-sets from the same writer
        v1.append(_set_vector(feats, pool_a, args.k))
        v2.append(_set_vector(feats, pool_a, args.k))
        labels.append(1.0)
        # negative: writer a vs a random different writer
        b = random.choice(writer_ids)
        while b == a and len(writer_ids) > 1:
            b = random.choice(writer_ids)
        pool_b = dataset._same_writer_pool(b)
        v1.append(_set_vector(feats, pool_a, args.k))
        v2.append(_set_vector(feats, pool_b, args.k))
        labels.append(0.0)
    v1 = torch.stack(v1)
    v2 = torch.stack(v2)
    labels = torch.tensor(labels, device=feats.device)
    return v1, v2, labels


def main():
    parser = argparse.ArgumentParser("train-writer-discriminator")
    parser.add_argument(
        "--data-dir", type=str,
        default="./saved_iam_data/combined_word_train",
        help="merged split directory built by utils/build_multidataset.py",
    )
    parser.add_argument(
        "--out", type=str, default="./style_models/writer_discriminator.pth",
        help="output path for the trained discriminator head",
    )
    parser.add_argument("--k", type=int, default=5, help="crops per set")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="anchors per step (each yields one positive + one negative pair); "
        "also the crop-encoding chunk size for the feature cache",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=200,
        help="pair-batches drawn per epoch",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    add_common_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = MergedWordDataset(args.data_dir, transforms=transform)
    writer_ids = sorted(dataset.writer_to_indices.keys())
    if len(writer_ids) < 2:
        raise RuntimeError("need >= 2 writers to train a discriminator")

    enc = load_extractor(args)
    feats = build_feature_cache(dataset, enc, args)
    print("feature cache", tuple(feats.shape))

    disc = WriterDiscriminator(feat_dim=feats.shape[1]).to(args.device)
    optimizer = optim.Adam(disc.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        disc.train()
        loss_meter = AvgMeter("bce")
        acc_meter = AvgMeter("acc")
        for _ in range(args.steps_per_epoch):
            v1, v2, labels = build_pair_batch(feats, dataset, writer_ids, args)
            logits = disc(v1, v2)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update((preds == labels).float().mean().item(), labels.size(0))

        print("epoch {}/{}  loss {:.4f}  acc {:.3f}".format(
            epoch + 1, args.epochs, loss_meter.avg, acc_meter.avg))

        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            torch.save(disc.state_dict(), args.out)
            print("saved best discriminator to", args.out)

    print("finished training; best loss", best_loss)


if __name__ == "__main__":
    main()
