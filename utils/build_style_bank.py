"""Build a precomputed per-writer style bank (stage-3 Part A).

For each IAM writer (class index 0..338) this encodes *all* of that writer's
word crops through the frozen ``ImageEncoder`` and stores the mean 1280-d vector,
producing ``saved_iam_data/style_bank.pt`` = a ``[339, 1280]`` float tensor.

The preprocessing (``iam_resizefix`` + ToTensor + Normalize) and the extractor
weights (``--style-path``) are identical to the live style path in
``models/diffpen2.py::get_style``, so a bank row is exactly the mean of the
per-crop features the UNet already consumes -- only lower-variance (mean over
all crops instead of 5 random) and reproducible.

Run on a GPU box:
    python -m utils.build_style_bank --device cuda:0 --style-path <style.pth>
"""

import argparse
import os

import torch
import torchvision
from torchvision import transforms

from models import ImageEncoder
from utils.iam_temploader import IAM_TempLoader, iam_resizefix
from utils.arghandle import add_common_args

NUM_WRITERS = 339
FEAT_DIM = 1280


def load_extractor(args):
    """Frozen mobilenet ImageEncoder with the trained style weights loaded
    (partial, shape-matched) -- mirrors utils/model_setup.py."""
    enc = ImageEncoder(
        model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True
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


def writer_crops(temp_loader, label_index):
    """Absolute paths of a writer's crops with transcription length > 3 (the
    same >3 filter the reference sampler uses); falls back to all crops."""
    wid = temp_loader.map_index_to_wid(label_index)
    rows = temp_loader.wmap.get(wid, [])
    paths = [
        os.path.join(temp_loader.root_path, r[0]) for r in rows if len(r[2]) > 3
    ]
    if not paths:
        paths = [os.path.join(temp_loader.root_path, r[0]) for r in rows]
    return paths


def encode_mean(paths, enc, transform, args, batch=64):
    """Mean feature vector over a writer's crops (streamed in batches)."""
    from PIL import Image

    total = torch.zeros(FEAT_DIM)
    count = 0
    buf = []

    def flush():
        nonlocal count, total, buf
        if not buf:
            return
        x = torch.stack(buf).to(args.device)
        with torch.no_grad():
            feat = enc(x).detach().cpu()
        total += feat.sum(dim=0)
        count += feat.shape[0]
        buf = []

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            print("skip unreadable crop:", p)
            continue
        buf.append(transform(iam_resizefix(img)))
        if len(buf) >= batch:
            flush()
    flush()
    if count == 0:
        return None
    return total / count


def main():
    parser = argparse.ArgumentParser("build-style-bank")
    parser.add_argument(
        "--out", type=str, default="./saved_iam_data/style_bank.pt",
        help="output path for the [339,1280] bank tensor",
    )
    add_common_args(parser)
    args = parser.parse_args()

    if args.dataset != "iam":
        raise ValueError("style bank build only supports the IAM dataset")

    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    IAM_TempLoader.check_preload()
    enc = load_extractor(args)

    bank = torch.zeros((NUM_WRITERS, FEAT_DIM))
    missing = []
    for idx in range(NUM_WRITERS):
        paths = writer_crops(IAM_TempLoader, idx)
        mean = encode_mean(paths, enc, transform, args)
        if mean is None:
            missing.append(idx)
            continue
        bank[idx] = mean
        if idx % 25 == 0:
            print(f"writer {idx}/{NUM_WRITERS}: {len(paths)} crops")

    if missing:
        print("WARNING: no crops for writer indices", missing)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(bank, args.out)
    print("saved style bank", args.out, tuple(bank.shape))


if __name__ == "__main__":
    main()
