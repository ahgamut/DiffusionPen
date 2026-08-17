"""Build the placer memmap dataset dir from the existing pipeline.

Word-level IAM/CVL/CSAFE datasets are now built by ``utils/build_multidataset.py``
(one merged memmap split). This module only covers the **placer** dataset (IAM
paragraph layout), which is a separate modality:

- **convert** (default): read the existing ``placer_IAM.pt`` cache and re-emit it
  as a memmap split.
- **rebuild**: construct ``IAMPlacerDataset`` (which parses the raw IAM xml/forms
  and writes ``placer_IAM.pt``), then convert from that.

    python -m utils.build_dataset            # convert placer_IAM.pt
    python -m utils.build_dataset --rebuild  # parse raw IAM first
"""

import argparse
import os

SAVE_ROOT = "./saved_iam_data"


def placer_dir(root):
    return os.path.join(root, "iam_placer")


def _placer_raw(rebuild):
    import torch

    pt = os.path.join(SAVE_ROOT, "placer_IAM.pt")
    if os.path.isfile(pt) and not rebuild:
        print("convert from", pt)
        return torch.load(pt, weights_only=False), os.path.basename(pt)

    print("rebuild placer from raw IAM xml/forms")
    from utils.placer_iam import IAMPlacerDataset

    IAMPlacerDataset()  # builds placer_IAM.pt
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


def main():
    parser = argparse.ArgumentParser("build-dataset")
    parser.add_argument("--out-root", default=SAVE_ROOT)
    parser.add_argument(
        "--rebuild", action="store_true", help="parse raw IAM instead of converting a .pt"
    )
    args = parser.parse_args()
    build_placer(args.out_root, args.rebuild)


if __name__ == "__main__":
    main()
