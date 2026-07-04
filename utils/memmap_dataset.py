"""Raw-memmap + msgpack dataset format (stage-4 dataset-format swap).

One directory per split holds:
- ``images.npy``   -- ``np.lib.format.open_memmap`` array ``(N, 64, 256, 3)`` uint8,
                      RGB, self-describing header, fork-safe read-only mmap.
- ``meta.msgpack`` -- ``list[dict]`` length N (per-crop labels/geometry; no pixels).
- ``index.msgpack``-- ``{by_writer: {wid: [rows]}, sequences: [[rows]], pairs: [[c,n]]}``
                      (whatever a given dataset needs; missing keys are fine).
- ``manifest.json``-- ``{version, N, shape, dtype, channels, subset, built_from}``.

Design goals: zero-decode pixel reads, O(1) same-writer sampling (from the index),
and a byte-identical drop-in for the existing ``.pt`` caches. ``msgpack`` is imported
lazily (only when this module is actually used) so the legacy ``.pt`` code path keeps
working without the new dependency installed.
"""

import json
import os

import numpy as np

FORMAT_VERSION = 1
IMG_H, IMG_W, IMG_C = 64, 256, 3
IMG_SHAPE = (IMG_H, IMG_W, IMG_C)

IMAGES_FILE = "images.npy"
META_FILE = "meta.msgpack"
INDEX_FILE = "index.msgpack"
MANIFEST_FILE = "manifest.json"


def _msgpack():
    # lazy: only needed when the memmap format is actually touched, so the
    # legacy .pt path keeps working without msgpack installed.
    try:
        import msgpack
    except ImportError as e:
        raise ImportError(
            "the stage-4 memmap dataset format needs msgpack "
            "(`pip install msgpack`; see requirements.txt)"
        ) from e
    return msgpack


def to_uint8_hwc(img):
    """Coerce a PIL image / raw ``tobytes()`` blob / ndarray into a
    ``(64,256,3)`` uint8 RGB array. Raw bytes are assumed row-major RGB from a
    256x64 (WxH) PIL image, matching ``Image.frombytes(mode='RGB', size=(256,64))``."""
    if isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(bytes(img), dtype=np.uint8).reshape(IMG_SHAPE)
        return arr
    if isinstance(img, np.ndarray):
        return np.ascontiguousarray(img.astype(np.uint8))
    # assume PIL.Image
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if arr.shape != IMG_SHAPE:
        raise ValueError(f"expected {IMG_SHAPE} image, got {arr.shape}")
    return arr


def split_exists(dirpath):
    return os.path.isfile(os.path.join(dirpath, IMAGES_FILE)) and os.path.isfile(
        os.path.join(dirpath, META_FILE)
    )


class MemmapWriter:
    """Streams crops straight to an ``open_memmap`` array (bounded memory), then
    dumps the metadata/index/manifest. Usage::

        w = MemmapWriter(dirpath, N)
        for row, crop in enumerate(...):
            w.write_image(row, crop)
        w.finalize(meta, index, built_from="placer_IAM.pt", subset="train")
    """

    def __init__(self, dirpath, n, shape=IMG_SHAPE):
        os.makedirs(dirpath, exist_ok=True)
        self.dirpath = dirpath
        self.n = n
        self.shape = shape
        self.images = np.lib.format.open_memmap(
            os.path.join(dirpath, IMAGES_FILE),
            mode="w+",
            dtype=np.uint8,
            shape=(n,) + tuple(shape),
        )

    def write_image(self, row, img):
        self.images[row] = to_uint8_hwc(img)

    def finalize(self, meta, index=None, built_from="", subset=""):
        assert len(meta) == self.n, f"meta len {len(meta)} != N {self.n}"
        self.images.flush()
        msgpack = _msgpack()
        with open(os.path.join(self.dirpath, META_FILE), "wb") as f:
            f.write(msgpack.packb(meta, use_bin_type=True))
        with open(os.path.join(self.dirpath, INDEX_FILE), "wb") as f:
            f.write(msgpack.packb(index or {}, use_bin_type=True))
        manifest = {
            "version": FORMAT_VERSION,
            "N": self.n,
            "shape": list(self.shape),
            "dtype": "uint8",
            "channels": self.shape[-1],
            "subset": subset,
            "built_from": built_from,
        }
        with open(os.path.join(self.dirpath, MANIFEST_FILE), "w") as f:
            json.dump(manifest, f)
        print("wrote memmap split", self.dirpath, "N=", self.n)


def load_images(dirpath):
    """Read-only, fork-safe (multi-worker) memmap of the pixel array."""
    return np.load(os.path.join(dirpath, IMAGES_FILE), mmap_mode="r")


def load_meta(dirpath):
    msgpack = _msgpack()
    with open(os.path.join(dirpath, META_FILE), "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)


def load_index(dirpath):
    msgpack = _msgpack()
    with open(os.path.join(dirpath, INDEX_FILE), "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)


def load_manifest(dirpath):
    with open(os.path.join(dirpath, MANIFEST_FILE), "r") as f:
        return json.load(f)


class WimgsView:
    """Sequence adapter exposing ``wimgs[i] -> RGB bytes`` over a memmap image
    array, so byte-consumers (`Image.frombytes`, upsampler) stay drop-in."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return np.ascontiguousarray(self.images[i]).tobytes()
