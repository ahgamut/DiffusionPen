"""Benchmark the siamese writer discriminator (evaluation statistics).

Given a merged split (the training set OR any held-out / external split) and a
trained discriminator head, sample K crops per writer, build style vectors from
the frozen extractor, and score writer comparisons two ways: raw MSE between
style vectors, and the discriminator's same-writer probability.

Emits one CSV row per comparison ``(writer_a, writer_b, k, mse_score,
linear_score)``: for each writer a self-comparison (writer vs itself, two
independent K-sets) plus ``--num-diff`` comparisons against random different
writers. K-sets need not be disjoint.

    python benchmark_disc.py --data-dir saved_iam_data/combined_word_test \\
        --discriminator-path ./style_models/writer_discriminator.pth \\
        --style-name resnet18 --style-path <style.pth> --device cuda:0 \\
        --k 5 --num-diff 5 --out ./saved_iam_data/disc_benchmark.csv
"""

import argparse
import csv
import os
import random

import torch
from torchvision import transforms

from models import WriterDiscriminator
from utils.word_dataset import MergedWordDataset, _sample
from utils.build_style_bank import load_extractor, build_feature_cache
from utils.arghandle import add_common_args


def _set_vector(feats, pool, k):
    return feats[_sample(pool, k)].mean(dim=0)


def _scores(disc, va, vb):
    mse = ((va - vb) ** 2).mean().item()
    with torch.no_grad():
        linear = torch.sigmoid(disc(va.unsqueeze(0), vb.unsqueeze(0))).item()
    return mse, linear


def main():
    parser = argparse.ArgumentParser("benchmark-writer-discriminator")
    parser.add_argument(
        "--data-dir", type=str,
        default="./saved_iam_data/combined_word_test",
        help="merged split to benchmark (training set or a held-out split)",
    )
    parser.add_argument(
        "--discriminator-path", type=str,
        default="./style_models/writer_discriminator.pth",
    )
    parser.add_argument(
        "--out", type=str, default="./saved_iam_data/disc_benchmark.csv",
    )
    parser.add_argument("--k", type=int, default=5, help="crops per set")
    parser.add_argument(
        "--num-diff", type=int, default=5,
        help="random different writers compared against each writer",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="crop-encoding chunk size for the feature cache",
    )
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
    if not writer_ids:
        raise RuntimeError("no writers in split " + args.data_dir)

    enc = load_extractor(args)
    feats = build_feature_cache(dataset, enc, args)
    print("feature cache", tuple(feats.shape))

    disc = WriterDiscriminator(feat_dim=feats.shape[1]).to(args.device)
    disc.load_state_dict(
        torch.load(args.discriminator_path, map_location=args.device, weights_only=True)
    )
    disc.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["writer_a", "writer_b", "k", "mse_score", "linear_score"])
        for a in writer_ids:
            pool_a = dataset._same_writer_pool(a)
            va = _set_vector(feats, pool_a, args.k)

            # self comparison (positive): a second independent K-set from a
            va2 = _set_vector(feats, pool_a, args.k)
            mse, linear = _scores(disc, va, va2)
            writer.writerow([a, a, args.k, mse, linear])

            # different-writer comparisons (negative)
            others = [w for w in writer_ids if w != a]
            random.shuffle(others)
            for b in others[:args.num_diff]:
                vb = _set_vector(feats, dataset._same_writer_pool(b), args.k)
                mse, linear = _scores(disc, va, vb)
                writer.writerow([a, b, args.k, mse, linear])

    print("wrote benchmark CSV to", args.out)


if __name__ == "__main__":
    main()
