"""Unified page-prompt loaders for IAM / CVL / CSAFE (replaced-paragraph eval).

Each dataset ships full-page images plus per-word bounding boxes in a different
XML dialect. This module normalizes all three to one minimal ``Prompt``-like
interface -- a page image path, per-word bboxes with transcription, and a
paragraph crop region -- which is exactly what
``utils.generation.build_replaced_paragraph`` consumes. So the same in-place
word-replacement path works for every dataset.

- **IAM**   -- form-XML (``xml/<form>.xml``) parsed by ``utils.subprompt.Prompt``;
              page is ``forms/<idd>.png``. Reused as-is (already this interface).
- **CVL**   -- PRImA PAGE XML (``xml/<page>_attributes.xml``): word regions are
              ``AttrRegion attrType="1"`` carrying ``text`` + a direct
              ``minAreaRect`` (4 ``Point``s -> axis-aligned bbox); page is
              ``pages/<imageFilename>.tif``. NB the files declare ``UTF-16`` but
              are actually ASCII, so the declaration is stripped before parsing.
- **CSAFE** -- Pascal-VOC (``w####_*.xml``): each accepted ``<object>`` gives a
              ``<name>`` + ``<bndbox>``; page is the sibling ``<stem>.png``.

Writer identity for the generated swaps is the **global** writer id from the
merged split's ``writers_global.json`` (``WriterIndex``), so it indexes a style
bank built over that merged split -- the same bank the diffusion model samples
from (see stage-3 style bank + ``--allow-bank-mismatch``).
"""

import glob
import json
import os
import re
import xml.etree.ElementTree as ET

from utils.subprompt import Prompt as IAMPrompt

# dataset key -> the prefix used in writers_global.json ("<DATASET>/<writer>")
DATASET_KEY = {"iam": "IAM", "cvl": "CVL", "csafe": "CSAFE"}
DEFAULT_ROOT = {"iam": "./iam_data", "cvl": "./cvl_data", "csafe": "./csafe_data"}


def _ln(tag):
    """Local (namespace-stripped) XML tag name."""
    return tag.split("}")[-1]


def _parse_relaxed(path):
    """Parse XML tolerant of a wrong/`UTF-16` encoding declaration: decode the
    bytes ourselves and drop the declaration so ElementTree can't object."""
    text = open(path, "rb").read().decode("utf-8", "replace")
    text = re.sub(r"<\?xml[^>]*\?>", "", text, count=1)
    return ET.fromstring(text)


class PageWord:
    """A word with its page-absolute bbox and transcription."""

    __slots__ = ("x_start", "y_start", "x_end", "y_end", "raw")

    def __init__(self, x_start, y_start, x_end, y_end, raw):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.raw = raw

    @property
    def width(self):
        return self.x_end - self.x_start

    @property
    def height(self):
        return self.y_end - self.y_start


class PagePrompt:
    """Minimal ``Prompt``-compatible container: page dims + per-word bboxes +
    writer + paragraph crop, enough for ``build_replaced_paragraph``."""

    def __init__(self, idd, writer_id, page_path, img_width, img_height, words, pad=10):
        self.idd = idd
        self.writer_id = writer_id
        self.page_path = page_path
        self.img_width = img_width
        self.img_height = img_height
        self.words = words
        # paragraph region = union of word bboxes, padded and clamped to the page
        x0 = min(w.x_start for w in words)
        y0 = min(w.y_start for w in words)
        x1 = max(w.x_end for w in words)
        y1 = max(w.y_end for w in words)
        self.x_start = max(0, x0 - pad)
        self.y_start = max(0, y0 - pad)
        self.x_end = min(img_width, x1 + pad)
        self.y_end = min(img_height, y1 + pad)
        self.width = self.x_end - self.x_start
        self.height = self.y_end - self.y_start

    def get_cropped(self, img):
        return img.crop((self.x_start, self.y_start, self.x_end, self.y_end))


# --------------------------------------------------------------------------- #
# per-dataset loaders: (list files) + (load one -> Prompt-like, or None)
# --------------------------------------------------------------------------- #
def _iam_files(root):
    return sorted(glob.glob(os.path.join(root, "xml", "*.xml")))


def _load_iam(path, root):
    p = IAMPrompt(path)
    # IAMPrompt already exposes idd/writer_id/words/img_*/get_cropped; add the
    # page path so the generalized caller opens it like the others.
    p.page_path = os.path.join(root, "forms", p.idd + ".png")
    return p


def _csafe_files(root):
    return sorted(glob.glob(os.path.join(root, "*.xml")))


def _load_csafe(path, root):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"w(\d{4})", stem)
    if not m:
        return None
    page = os.path.join(root, stem + ".png")
    if not os.path.isfile(page):
        return None
    r = _parse_relaxed(path)
    size = r.find("size")
    img_w = int(float(size.findtext("width")))
    img_h = int(float(size.findtext("height")))
    words = []
    for obj in r.iter("object"):
        if obj.findtext("status") != "accepted":
            continue
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        if not name or bb is None:
            continue
        try:
            x0 = int(round(float(bb.findtext("xmin"))))
            y0 = int(round(float(bb.findtext("ymin"))))
            x1 = int(round(float(bb.findtext("xmax"))))
            y1 = int(round(float(bb.findtext("ymax"))))
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        words.append(PageWord(x0, y0, x1, y1, name))
    if not words:
        return None
    return PagePrompt(stem, m.group(1), page, img_w, img_h, words)


def _cvl_files(root):
    return sorted(glob.glob(os.path.join(root, "xml", "*_attributes.xml")))


def _cvl_bbox(region):
    """Axis-aligned bbox from a word region's direct ``minAreaRect`` points."""
    mar = [c for c in region if _ln(c.tag) == "minAreaRect"]
    if not mar:
        return None
    pts = [
        (float(p.attrib["x"]), float(p.attrib["y"]))
        for p in mar[0]
        if _ln(p.tag) == "Point"
    ]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _load_cvl(path, root):
    r = _parse_relaxed(path)
    page_el = next((e for e in r.iter() if _ln(e.tag) == "Page"), None)
    if page_el is None:
        return None
    img_name = page_el.attrib.get("imageFilename")
    try:
        img_w = int(float(page_el.attrib["imageWidth"]))
        img_h = int(float(page_el.attrib["imageHeight"]))
    except (KeyError, TypeError, ValueError):
        return None
    page = os.path.join(root, "pages", img_name)
    if not os.path.isfile(page):
        return None
    idd = os.path.splitext(img_name)[0]  # e.g. "0001-1"
    writer = idd.split("-")[0]  # e.g. "0001"
    words = []
    for e in r.iter():
        # word regions are attrType="1" AttrRegions that carry a text attribute
        # (the geometry-only attrType="1" glyph pass has no text).
        if _ln(e.tag) != "AttrRegion":
            continue
        if e.attrib.get("attrType") != "1" or e.attrib.get("text") is None:
            continue
        bb = _cvl_bbox(e)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        x0 = max(0, min(int(round(x0)), img_w))
        x1 = max(0, min(int(round(x1)), img_w))
        y0 = max(0, min(int(round(y0)), img_h))
        y1 = max(0, min(int(round(y1)), img_h))
        if x1 <= x0 or y1 <= y0:
            continue
        words.append(PageWord(x0, y0, x1, y1, e.attrib["text"]))
    if not words:
        return None
    return PagePrompt(idd, writer, page, img_w, img_h, words)


_LOADERS = {
    "iam": (_iam_files, _load_iam),
    "cvl": (_cvl_files, _load_cvl),
    "csafe": (_csafe_files, _load_csafe),
}


def collect_prompt_files(dataset, root):
    """All candidate prompt-XML paths for a dataset under ``root``."""
    return _LOADERS[dataset][0](root)


def load_page_prompt(dataset, path, root):
    """Load one prompt as a ``Prompt``-like object (or None on parse failure)."""
    return _LOADERS[dataset][1](path, root)


class WriterIndex:
    """Map a dataset's raw writer to the global writer id used by the merged
    style bank, via a merged split's ``writers_global.json``."""

    def __init__(self, writers_global_path):
        reg = json.load(open(writers_global_path))
        self.map = reg["writer_to_wid"]
        self.path = writers_global_path

    def index(self, dataset, raw_writer):
        key = "{}/{}".format(DATASET_KEY[dataset], raw_writer)
        if key not in self.map:
            raise KeyError(
                "writer {!r} not in {}; make sure this is the writers_global.json "
                "of the merged split the style bank was built from".format(
                    key, self.path
                )
            )
        return int(self.map[key])
