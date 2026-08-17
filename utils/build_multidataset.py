"""Build one merged memmap split from bespoke IAM / CVL / CSAFE sample folders.

Point ``--input`` at a folder that contains any of the subfolders ``IAM/``,
``CVL/``, ``CSAFE/`` (the same layout as ``./sample-fmt/``). The three datasets
ship in different on-disk shapes:

- **IAM**   -- ``xml/<form>.xml`` gives ``<form writer-id>`` + per-``<word>`` text;
              the crop is the matching pre-cut ``words/aXX/aXX-XXXu/<word-id>.png``.
- **CVL**   -- pre-cut ``words/<wid>/<wid>-<pg>-<ln>-<wd>-<TEXT>.tif``; writer and
              text come straight from the filename (``xml/`` locations unused).
- **CSAFE** -- flat ``w<wid>_..._.png`` pages + Pascal-VOC ``.xml`` (no pre-cut
              crops): each ``<object status="accepted">`` is cropped from the page
              by its ``<bndbox>``; writer is the 4 digits after the leading ``w``.

Output is a single stage-4 memmap split dir (see ``utils/memmap_dataset.py``):
``images.npy`` + ``meta.msgpack`` + ``index.msgpack`` + ``manifest.json``, plus a
``writers_global.json`` registry written inside it. Every crop's metadata carries
its source ``dataset`` and raw ``writer`` plus a **globally-namespaced integer
writer id** (``wid``, contiguous ``0..W-1`` over the sorted ``(dataset, writer)``
pairs), so writers from different datasets never collide -- IAM ``000``, CVL
``0001`` and CSAFE ``0001`` get distinct ids.

    python -m utils.build_multidataset --input ./sample-fmt --split-name train

Deterministic (sorted writer keys, no RNG) -> reproducible ids. Crops are
normalized to the same 256x64 RGB geometry the existing IAM/CVL loaders produce,
so the merged split is byte-compatible with the stage-4 word-level format.
"""

import argparse
import glob
import json
import os
import re
import string
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps

from utils.auxiliary_functions import centered_PIL, image_resize_PIL
from utils.memmap_dataset import MemmapWriter

IMG_H, IMG_W = 64, 256
# subdir name under --input for each dataset key
DATASET_DIRS = {"iam": "IAM", "cvl": "CVL", "csafe": "CSAFE"}


# --------------------------------------------------------------------------- #
# crop normalization (kept identical to the existing IAM/CVL word-crop path)
# --------------------------------------------------------------------------- #
def normalize_word_crop(img, transcr):
    """Coerce a word image to a 256x64 RGB ``PIL.Image``.

    Mirrors ``utils/iam_dataset.py`` / ``utils/cvl_dataset.py``: resize to height
    64 keeping aspect; if the result is still wider than 256, shrink it ~20px at a
    time and center it; otherwise white-pad it to 256 wide. Punctuation-only words
    are centered directly. (The two existing loaders differ only trivially in the
    pad-vs-shrink ordering; this is the canonical order used for all datasets.)
    """
    img = img.convert("RGB")
    if transcr in string.punctuation:
        return centered_PIL(img, (IMG_H, IMG_W), border_value=255.0)

    w, h = img.size
    img = img.resize((max(1, int(w * IMG_H / h)), IMG_H))
    w, h = img.size
    if w < IMG_W:
        return ImageOps.pad(img, size=(IMG_W, IMG_H), color="white")
    while w > IMG_W:
        img = image_resize_PIL(img, width=w - 20)
        w, h = img.size
    return centered_PIL(img, (IMG_H, IMG_W), border_value=255.0)


# --------------------------------------------------------------------------- #
# per-dataset adapters -> list[record]
#   record = {dataset, writer, transcr, ref}
#   ref = ("crop", path)  |  ("bbox", page_path, (left, top, right, bottom))
# All validation (missing file / empty text / degenerate bbox) happens HERE so
# that N is exact and the single write pass below never has to drop a row.
# --------------------------------------------------------------------------- #
def iam_records(root, stats, keep_nonok=False):
    """IAM: one record per ``<word>`` that has a matching pre-cut crop png.

    By default only words in lines marked ``segmentation="ok"`` are kept (IAM's
    own quality marker, matching the existing loader's intent). ``keep_nonok``
    includes words from poorly-segmented lines too -- their crops still exist on
    disk, so pass this to maximize data volume at some quality cost.
    """
    xml_dir = os.path.join(root, "xml")
    words_dir = os.path.join(root, "words")
    recs = []
    for xf in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        try:
            form = ET.parse(xf).getroot()
        except ET.ParseError as e:
            print("  [IAM] skip unparseable xml", xf, e)
            stats["bad_xml"] += 1
            continue
        raw_writer = form.get("writer-id")
        if raw_writer is None:
            print("  [IAM] skip form without writer-id", xf)
            stats["no_writer"] += 1
            continue
        for line in form.iter("line"):
            # keep only well-segmented lines unless --iam-keep-nonok; tolerate
            # lines that omit the attribute entirely.
            if not keep_nonok and line.get("segmentation") not in (None, "ok"):
                stats["nonok_line"] += 1
                continue
            for word in line.iter("word"):
                wid_str = word.get("id")
                transcr = word.get("text")
                if not wid_str or not transcr:
                    stats["empty_text"] += 1
                    continue
                parts = wid_str.split("-")  # e.g. a01-000u-00-00
                if len(parts) < 2:
                    continue
                crop = os.path.join(
                    words_dir, parts[0], "-".join(parts[:2]), wid_str + ".png"
                )
                if not os.path.isfile(crop):
                    stats["missing_crop"] += 1
                    continue
                recs.append(
                    {
                        "dataset": "IAM",
                        "writer": raw_writer,
                        "transcr": transcr,
                        "ref": ("crop", crop),
                    }
                )
    return recs


def cvl_records(root, stats):
    """CVL: one record per pre-cut word tif; writer/text from the filename."""
    recs = []
    for tif in sorted(glob.glob(os.path.join(root, "words", "*", "*.tif"))):
        stem = os.path.splitext(os.path.basename(tif))[0]
        tok = stem.split("-")  # <wid>-<pg>-<ln>-<wd>-<TEXT...>
        if len(tok) < 5:
            print("  [CVL] skip unparseable filename", tif)
            stats["bad_name"] += 1
            continue
        raw_writer = tok[0]
        transcr = "-".join(tok[4:])  # rejoin: words may contain hyphens
        if not transcr:
            stats["empty_text"] += 1
            continue
        recs.append(
            {
                "dataset": "CVL",
                "writer": raw_writer,
                "transcr": transcr,
                "ref": ("crop", tif),
            }
        )
    return recs


def _clamp_box(box, size):
    """Clamp (l,t,r,b) into a WxH page; return None if the area is degenerate."""
    l, t, r, b = box
    if size is not None:
        pw, ph = size
        l, r = max(0, min(l, pw)), max(0, min(r, pw))
        t, b = max(0, min(t, ph)), max(0, min(b, ph))
    if r <= l or b <= t:
        return None
    return (l, t, r, b)


def csafe_records(root, stats):
    """CSAFE: crop each accepted <object> from its page via the bndbox."""
    recs = []
    for xf in sorted(glob.glob(os.path.join(root, "*.xml"))):
        stem = os.path.splitext(os.path.basename(xf))[0]
        m = re.match(r"w(\d{4})", stem)
        if not m:
            print("  [CSAFE] skip file without w#### writer", xf)
            stats["no_writer"] += 1
            continue
        raw_writer = m.group(1)
        page = os.path.join(root, stem + ".png")
        if not os.path.isfile(page):
            print("  [CSAFE] skip missing page png for", xf)
            stats["missing_page"] += 1
            continue
        try:
            root_el = ET.parse(xf).getroot()
        except ET.ParseError as e:
            print("  [CSAFE] skip unparseable xml", xf, e)
            stats["bad_xml"] += 1
            continue
        # page dims from the xml <size> (present in this format) -> validate bbox
        # without opening the png; fall back to None (clamp against the real image
        # in the write pass) if absent.
        size_el = root_el.find("size")
        page_size = None
        if size_el is not None:
            try:
                page_size = (
                    int(float(size_el.findtext("width"))),
                    int(float(size_el.findtext("height"))),
                )
            except (TypeError, ValueError):
                page_size = None
        for obj in root_el.iter("object"):
            if obj.findtext("status") != "accepted":
                stats["not_accepted"] += 1
                continue
            name = obj.findtext("name")
            if not name:
                stats["empty_text"] += 1
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                box = (
                    int(round(float(bb.findtext("xmin")))),
                    int(round(float(bb.findtext("ymin")))),
                    int(round(float(bb.findtext("xmax")))),
                    int(round(float(bb.findtext("ymax")))),
                )
            except (TypeError, ValueError):
                continue
            box = _clamp_box(box, page_size)
            if box is None:
                stats["bad_bbox"] += 1
                continue
            recs.append(
                {
                    "dataset": "CSAFE",
                    "writer": raw_writer,
                    "transcr": name,
                    "ref": ("bbox", page, box),
                }
            )
    return recs


ADAPTERS = {"iam": iam_records, "cvl": cvl_records, "csafe": csafe_records}


# --------------------------------------------------------------------------- #
# write pass
# --------------------------------------------------------------------------- #
def load_crop(ref, page_cache):
    """Materialize a record's raw crop as a PIL image. CSAFE pages are cached
    (records are page-contiguous, so a single-entry cache opens each page once)."""
    if ref[0] == "crop":
        return Image.open(ref[1])
    _, page_path, box = ref
    page = page_cache.get(page_path)
    if page is None:
        page = Image.open(page_path).convert("RGB")
        page_cache.clear()  # bounded to one page in memory
        page_cache[page_path] = page
    box = _clamp_box(box, page.size)
    if box is None:
        return None
    return page.crop(box)


def build(input_root, split_name, out_root, out_name, datasets, iam_keep_nonok=False):
    # ---- pass 1: enumerate + validate (no pixels loaded yet) -------------- #
    records = []
    stats = {}
    for key in datasets:
        sub = os.path.join(input_root, DATASET_DIRS[key])
        if not os.path.isdir(sub):
            print(f"[{key}] no {DATASET_DIRS[key]}/ under {input_root}; skipping")
            continue
        s = _new_stats()
        if key == "iam":
            recs = iam_records(sub, s, keep_nonok=iam_keep_nonok)
        else:
            recs = ADAPTERS[key](sub, s)
        stats[key] = s
        print(f"[{key}] {len(recs)} records  (skips: {_fmt_stats(s)})")
        records.extend(recs)

    if not records:
        raise SystemExit(f"no usable records found under {input_root}")

    # ---- global, deterministic writer registry --------------------------- #
    keys = sorted({(r["dataset"], r["writer"]) for r in records})
    writer_to_wid = {f"{d}/{w}": i for i, (d, w) in enumerate(keys)}
    n_writers = len(keys)
    n = len(records)
    print(f"total: N={n} crops, W={n_writers} writers")

    out_dir = os.path.join(out_root, f"{out_name}_{split_name}")

    # ---- pass 2: materialize crops -> memmap ----------------------------- #
    writer = MemmapWriter(out_dir, n)
    meta = []
    by_writer, by_writer_long = {}, {}
    page_cache = {}
    for row, r in enumerate(records):
        crop = load_crop(r["ref"], page_cache)
        if crop is None:
            # pass 1 already validated existence/geometry; a failure here is
            # exceptional (e.g. a truncated file) -> fail loud, don't desync N.
            raise RuntimeError(f"could not materialize crop for row {row}: {r['ref']}")
        writer.write_image(row, normalize_word_crop(crop, r["transcr"]))

        wid = writer_to_wid[f"{r['dataset']}/{r['writer']}"]
        src_id = r["ref"][1]  # source crop path or page path
        meta.append(
            {
                "transcr": r["transcr"],
                "wid": wid,
                "id": src_id,
                "dataset": r["dataset"],
                "writer": r["writer"],
            }
        )
        by_writer.setdefault(str(wid), []).append(row)
        if len(r["transcr"]) > 3:
            by_writer_long.setdefault(str(wid), []).append(row)
        if row % 1000 == 0:
            print(f"  wrote {row}/{n}")

    index = {"by_writer": by_writer, "by_writer_long": by_writer_long}
    writer.finalize(meta, index, built_from="multidataset", subset=split_name)

    # writer registry travels inside the split dir
    reg = {
        "writer_to_wid": writer_to_wid,
        "wid_to_writer": {str(v): k for k, v in writer_to_wid.items()},
        "n_writers": n_writers,
    }
    with open(os.path.join(out_dir, "writers_global.json"), "w") as f:
        json.dump(reg, f, indent=2)
    print(f"done: {out_dir}  (N={n}, W={n_writers})")


# --------------------------------------------------------------------------- #
# skip-stat helpers
# --------------------------------------------------------------------------- #
_STAT_KEYS = (
    "bad_xml",
    "no_writer",
    "empty_text",
    "missing_crop",
    "nonok_line",
    "bad_name",
    "missing_page",
    "bad_bbox",
    "not_accepted",
)


def _new_stats():
    return {k: 0 for k in _STAT_KEYS}


def _fmt_stats(s):
    hit = {k: v for k, v in s.items() if v}
    return ", ".join(f"{k}={v}" for k, v in hit.items()) or "none"


def main():
    p = argparse.ArgumentParser("build-multidataset")
    p.add_argument(
        "--input", required=True,
        help="folder containing IAM/ CVL/ CSAFE/ subdirs (like ./sample-fmt)",
    )
    p.add_argument("--split-name", default="train", help="output split name suffix")
    p.add_argument("--out-root", default="./saved_iam_data")
    p.add_argument("--out-name", default="combined_word", help="output dir prefix")
    p.add_argument(
        "--datasets", default="iam,cvl,csafe",
        help="comma-separated subset of {iam,cvl,csafe} to include",
    )
    p.add_argument(
        "--iam-keep-nonok", action="store_true",
        help="also keep IAM words from lines whose segmentation != 'ok' "
        "(their crops exist on disk; default drops them for quality)",
    )
    args = p.parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    bad = [d for d in datasets if d not in DATASET_DIRS]
    if bad:
        raise SystemExit(f"unknown dataset(s): {bad}; choose from {list(DATASET_DIRS)}")

    build(
        args.input, args.split_name, args.out_root, args.out_name, datasets,
        iam_keep_nonok=args.iam_keep_nonok,
    )


if __name__ == "__main__":
    main()
