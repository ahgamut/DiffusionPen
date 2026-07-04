"""Build the stage-4 memmap dataset dirs from the existing pipeline.

Two sources, one memmap sink (utils/memmap_dataset.py):
- **convert** (default, fast, no raw tree): read an existing ``.pt`` cache and
  re-emit it as a memmap split.
- **rebuild**: construct the dataset object (which runs the existing parsing and
  writes the ``.pt``), then convert from its in-memory ``data``.

Covers all three datasets: word-level IAM (train.py), placer (placer_IAM.pt),
and CVL. Reuses the existing parsing entirely -- only the serialization sink
changes -- so crops/labels are byte-identical to the ``.pt`` path.

    python -m utils.build_dataset --dataset all --subset train
    python -m utils.build_dataset --dataset word --subset test --rebuild
"""

import argparse
import os
from types import SimpleNamespace

SAVE_ROOT = "./saved_iam_data"


# ----- directory naming (consumers auto-detect these) -----
def word_dir(root, subset):
    return os.path.join(root, f"iam_word_{subset}")


def cvl_dir(root, subset):
    return os.path.join(root, f"cvl_word_{subset}")


def placer_dir(root):
    return os.path.join(root, "iam_placer")


# ----- word-level IAM -----
def _word_data(subset, rebuild):
    """Return list[(PIL, transcr, wid, path)] either from the .pt or by building."""
    import torch

    pt = os.path.join(SAVE_ROOT, f"{subset}_word_IAM.pt")
    if os.path.isfile(pt) and not rebuild:
        print("convert from", pt)
        return torch.load(pt, weights_only=False), os.path.basename(pt)

    print("rebuild word-level from raw IAM")
    from utils.iam_dataset import IAMDataset

    ds = IAMDataset(
        "./iam_data/words", subset, "word", (64, 256),
        None, None, None, None, SimpleNamespace(),
    )
    return ds.data, "raw:iam"


def build_word(subset, root, rebuild):
    from utils.memmap_dataset import MemmapWriter

    data, src = _word_data(subset, rebuild)
    n = len(data)
    w = MemmapWriter(word_dir(root, subset), n)
    meta = []
    by_writer, by_writer_long = {}, {}
    for row, (img, transcr, wid, path) in enumerate(data):
        w.write_image(row, img)
        meta.append({"transcr": transcr, "wid": int(wid), "id": path})
        by_writer.setdefault(str(int(wid)), []).append(row)
        if len(transcr) > 3:
            by_writer_long.setdefault(str(int(wid)), []).append(row)
    index = {"by_writer": by_writer, "by_writer_long": by_writer_long}
    w.finalize(meta, index, built_from=src, subset=subset)


# ----- placer (IAM paragraphs) -----
def _placer_raw(rebuild):
    import torch

    pt = os.path.join(SAVE_ROOT, "placer_IAM.pt")
    if os.path.isfile(pt) and not rebuild:
        print("convert from", pt)
        return torch.load(pt, weights_only=False), os.path.basename(pt)

    print("rebuild placer from raw IAM xml/forms")
    from utils.placer_iam import IAMPlacerDataset

    ds = IAMPlacerDataset()  # builds placer_IAM.pt
    return torch.load(pt, weights_only=False), "raw:iam-placer"


def build_placer(root, rebuild):
    from collections import OrderedDict

    from utils.memmap_dataset import MemmapWriter
    from utils.subprompt import Word
    from utils.placer_iam import RelWordIndices

    raw, src = _placer_raw(rebuild)
    words = [Word.from_bytes(x) for x in raw["words"]]
    wimgs = raw["wimgs"]
    pairs = [RelWordIndices.from_bytes(x) for x in raw["pairs"]]
    assert len(words) == len(wimgs), (len(words), len(wimgs))
    n = len(words)

    w = MemmapWriter(placer_dir(root), n)
    meta = []
    by_writer = {}
    docs = OrderedDict()
    for row, word in enumerate(words):
        w.write_image(row, wimgs[row])
        d = word.to_dict()
        d["row"] = row
        meta.append(d)
        by_writer.setdefault(str(word.writer_id), []).append(row)
        docs.setdefault(word.parent_doc, []).append(row)
    index = {
        "by_writer": by_writer,
        "sequences": list(docs.values()),
        "pairs": [[p.cur_index, p.next_index] for p in pairs],
    }
    w.finalize(meta, index, built_from=src, subset="all")


# ----- CVL -----
def _cvl_raw(subset, rebuild):
    import torch

    pt = os.path.join(SAVE_ROOT, f"{subset}_word_CVL.pt")
    if os.path.isfile(pt) and not rebuild:
        print("convert from", pt)
        return torch.load(pt, weights_only=False), os.path.basename(pt)

    print("rebuild CVL from raw")
    from utils.cvl_dataset import CVLDataset

    ds = CVLDataset(
        "./cvl_data", subset, "word", (64, 256),
        None, None, None, None, args=SimpleNamespace(),
    )
    return {"data": ds.data, "paths": ds.img_paths, "wmap": ds.wmap}, "raw:cvl"


def build_cvl(subset, root, rebuild):
    from utils.memmap_dataset import MemmapWriter

    raw, src = _cvl_raw(subset, rebuild)
    data, paths, wmap = raw["data"], raw["paths"], raw["wmap"]
    n = len(data)
    w = MemmapWriter(cvl_dir(root, subset), n)
    meta = []
    for row, (blob, transcr, wid) in enumerate(data):
        w.write_image(row, blob)
        meta.append({"transcr": transcr, "wid": wid, "id": paths[row]})
    # wmap is already {writer_id: [rows]} (O(1) sampling); keep string keys.
    index = {"by_writer": {str(k): list(v) for k, v in wmap.items()}}
    w.finalize(meta, index, built_from=src, subset=subset)


def main():
    parser = argparse.ArgumentParser("build-dataset")
    parser.add_argument(
        "--dataset", choices=["word", "placer", "cvl", "all"], default="all"
    )
    parser.add_argument("--subset", default="train", help="train/val/test (word, cvl)")
    parser.add_argument("--out-root", default=SAVE_ROOT)
    parser.add_argument(
        "--rebuild", action="store_true", help="parse raw instead of converting a .pt"
    )
    args = parser.parse_args()

    if args.dataset in ("word", "all"):
        build_word(args.subset, args.out_root, args.rebuild)
    if args.dataset in ("cvl", "all"):
        build_cvl(args.subset, args.out_root, args.rebuild)
    if args.dataset in ("placer", "all"):
        build_placer(args.out_root, args.rebuild)


if __name__ == "__main__":
    main()
