"""Build a precomputed per-writer style bank (stage-3 Part A).

For each writer in a merged word-dataset split this encodes the writer's word
crops through the frozen ``ImageEncoder`` and stores the mean feature vector,
producing a ``[W, feat]`` tensor where ``W`` is the split's writer count and
``feat`` is the encoder's output width (``mobilenetv2_100`` -> 1280,
``resnet18`` -> 512).

The crops, preprocessing (``dataset.transforms`` = ToTensor + Normalize) and the
extractor weights (``--style-path`` / ``--style-name``) are identical to the live
training/inference style path, so a bank row is exactly the mean of the per-crop
features the UNet consumes -- only lower-variance (mean over all of a writer's
crops instead of 5 random) and reproducible. Long-transcription crops (>3 chars)
are preferred, matching the same-writer reference sampler
(``MergedWordDataset._same_writer_pool``).

This reads the merged memmap split (built by ``utils.build_multidataset``) rather
than raw IAM, so it handles IAM/CVL/CSAFE and the global writer-id space; an
IAM-only split is just the ``W=339`` special case.

Run on a GPU box (build the merged split first):
    python -m utils.build_style_bank --data-dir saved_iam_data/combined_word_train \\
        --device cuda:0 --style-name resnet18 --style-path <style.pth>
"""

import argparse
import os

import torch
import torchvision
from torchvision import transforms

from models import ImageEncoder
from utils.word_dataset import MergedWordDataset
from utils.arghandle import add_common_args


def load_extractor(args):
    """Frozen ImageEncoder (``mobilenetv2_100`` or ``resnet18``) with the trained
    style weights partially (shape-matched) loaded -- mirrors the extractor built
    in utils/model_setup.py and train.py, sized by ``--style-name``."""
    if args.style_name not in ("mobilenetv2_100", "resnet18"):
        raise ValueError("unknown --style-name {!r}".format(args.style_name))
    enc = ImageEncoder(
        model_name=args.style_name, num_classes=0, pretrained=True, trainable=True
    )
    state = torch.load(args.style_path, map_location=args.device, weights_only=True)
    model_dict = enc.state_dict()
    state = {
        k: v
        for k, v in state.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(state)
    enc.load_state_dict(model_dict)
    enc = enc.to(args.device)
    enc.requires_grad_(False)
    enc.eval()
    return enc


def encode_mean(indices, dataset, enc, args):
    """Mean feature vector over a writer's crops, streamed in batches through the
    memmap-backed dataset with the same transforms training uses."""
    total = None
    count = 0
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start:start + args.batch_size]
            batch = torch.stack(
                [dataset.transforms(dataset._img(i)) for i in chunk]
            ).to(args.device)
            feat = enc(batch).detach().cpu()
            total = feat.sum(dim=0) if total is None else total + feat.sum(dim=0)
            count += feat.shape[0]
    if count == 0:
        return None
    return total / count


def main():
    parser = argparse.ArgumentParser("build-style-bank")
    parser.add_argument(
        "--data-dir", type=str,
        default="./saved_iam_data/combined_word_train",
        help="merged split directory built by utils/build_multidataset.py",
    )
    parser.add_argument(
        "--out", type=str, default="./saved_iam_data/style_bank.pt",
        help="output path for the [W, feat] bank tensor",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    add_common_args(parser)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = MergedWordDataset(args.data_dir, transforms=transform)
    # Size to the same writer count the model uses: an explicit --style-classes
    # override (matches model_setup), else the split's writer count W. The merged
    # builder assigns contiguous global ids 0..W-1, so wclasses == W == max+1.
    num_writers = int(getattr(args, "style_classes", 0) or 0) or dataset.wclasses
    enc = load_extractor(args)

    feats_by_writer = {}
    feat_dim = None
    missing = []
    writer_ids = sorted(dataset.writer_to_indices.keys())
    for n, wid in enumerate(writer_ids):
        idxs = dataset._same_writer_pool(wid)
        mean = encode_mean(idxs, dataset, enc, args)
        if mean is None:
            missing.append(wid)
            continue
        feats_by_writer[wid] = mean
        feat_dim = mean.numel()
        if n % 25 == 0:
            print("writer {}/{} (wid {}): {} crops".format(
                n, len(writer_ids), wid, len(idxs)))

    if feat_dim is None:
        raise RuntimeError("no writers encoded -- empty dataset at " + args.data_dir)

    bank = torch.zeros(num_writers, feat_dim)
    out_of_range = []
    for wid, vec in feats_by_writer.items():
        if 0 <= wid < num_writers:
            bank[wid] = vec.to(bank.dtype)
        else:
            out_of_range.append(wid)

    if out_of_range:
        print("WARNING: writer ids out of [0,{}) skipped: {}".format(
            num_writers, out_of_range))
    if missing:
        print("WARNING: no crops for writer ids", missing)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(bank, args.out)
    print("saved style bank", args.out, tuple(bank.shape),
          "| style-name", args.style_name, "| writers", num_writers)


if __name__ == "__main__":
    main()
