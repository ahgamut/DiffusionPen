"""Single training-data Dataset over the merged memmap format.

``MergedWordDataset`` is the one dataset class used for training in this
codebase. It reads only the stage-4 memmap split produced by
``utils/build_multidataset.py`` (IAM + CVL + CSAFE unified, writers in a global
id space); there is no legacy ``.pt`` / raw-parse fallback -- the split must
already exist on disk.

It serves both trainers via ``style_mode``:

- ``style_mode=False`` (default) -- the diffusion loop; ``__getitem__`` returns
  ``(img, transcr, wid, s_imgs[5], img_path, cor_im)``.
- ``style_mode=True`` -- style-encoder pretraining; ``__getitem__`` returns
  ``(img, transcr, wid, positive, negative, s_imgs[5])`` with a same-writer
  ``positive`` and a different-writer ``negative`` for the triplet loss.

Same-writer sampling is O(1) via the split's ``by_writer`` / ``by_writer_long``
index. The DataLoaders use the default collate (tensors stack, strings become
lists), so this class deliberately provides no ``collate_fn``.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import memmap_dataset as mm


def _sample(pool, k):
    """k indices from pool: without replacement when possible, else with
    replacement (merged writers vary in crop count, unlike the old IAM-only
    loader which assumed every writer had >= 5 crops)."""
    if len(pool) >= k:
        return random.sample(pool, k)
    return [random.choice(pool) for _ in range(k)]


class MergedWordDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        args=None,
        style_mode=False,
    ):
        self.transforms = transforms
        self.args = args
        self.style_mode = style_mode
        # the split directory itself; caches co-located with images.npy live here.
        self.data_dir = data_dir
        # basename of the split dir, used as a stable identity for cache keys
        # (e.g. train.py::build_style_cache keys on it).
        self.setname = os.path.basename(os.path.normpath(data_dir))
        # Optional [N, feat] tensor of precomputed frozen style-CNN features,
        # attached externally by train.py::build_style_cache; when set, the
        # diffusion path returns cached style vectors instead of raw crops.
        self.style_cache = None
        # Optional [N, 4, 8, 32] read-only float32 memmap of precomputed frozen
        # VAE latent means (unscaled), attached by train.py::build_latent_cache;
        # when set, the diffusion path returns the cached latent as the image, so
        # the training loop skips the per-step vae.encode. Memmapped so RAM stays
        # O(1) in N and forked workers share it.
        self.latent_cache = None

        mm_dir = data_dir
        if not mm.split_exists(mm_dir):
            raise RuntimeError(
                "merged split {} not found -- build it first, e.g.:\n"
                "  python -m utils.build_multidataset --input <folder> "
                "--out-name <name> --split-name <split>".format(mm_dir)
            )
        self.images = mm.load_images(mm_dir)  # read-only, fork-safe memmap
        meta = mm.load_meta(mm_dir)
        index = mm.load_index(mm_dir)
        # data rows keep the old (None, transcr, wid, id) shape; pixels live in
        # self.images and are fetched lazily via _img().
        self.data = [(None, m["transcr"], m["wid"], m.get("id", "")) for m in meta]
        print("loaded merged split", mm_dir, "N=", len(self.data))

        self.initial_writer_ids = [d[2] for d in self.data]
        self.writer_ids = [int(w) for w in np.unique(self.initial_writer_ids)]
        self.wclasses = len(self.writer_ids)
        print("Number of writers", self.wclasses)

        # writer -> row-indices for O(1) same-writer sampling (from the split's
        # index; "_long" keeps only transcriptions with len > 3).
        by_w = index.get("by_writer", {})
        by_wl = index.get("by_writer_long", {})
        self.writer_to_indices = {int(k): list(v) for k, v in by_w.items()}
        self.writer_to_indices_long = {int(k): list(v) for k, v in by_wl.items()}
        if not self.writer_to_indices:
            self._build_writer_index()

    def _build_writer_index(self):
        self.writer_to_indices = {}
        self.writer_to_indices_long = {}
        for i, p in enumerate(self.data):
            wid = p[2]
            self.writer_to_indices.setdefault(wid, []).append(i)
            if len(p[1]) > 3:
                self.writer_to_indices_long.setdefault(wid, []).append(i)

    def _img(self, index):
        """Word crop at ``index`` as a PIL image, from the memmap array."""
        return Image.fromarray(np.asarray(self.images[index]))

    def _same_writer_pool(self, wid):
        """Prefer long-transcription crops; fall back to all of the writer's."""
        return self.writer_to_indices_long.get(wid) or self.writer_to_indices[wid]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.style_mode:
            return self._getitem_style(index)
        return self._getitem_diffusion(index)

    def _getitem_diffusion(self, index):
        img_path = self.data[index][3]
        if self.latent_cache is not None:
            # precomputed VAE latent mean -> the loop skips vae.encode entirely.
            # np.array() copies the row out of the read-only memmap (writable,
            # no aliasing) before handing it to torch.
            img = torch.from_numpy(np.array(self.latent_cache[index])).float()
        else:
            img = self._img(index)
            if self.transforms is not None:
                img = self.transforms(img)
        transcr = self.data[index][1]
        wid = self.data[index][2]

        pool = self._same_writer_pool(wid)
        random_samples = _sample(pool, 5)
        cor_im = self.transforms(self._img(random.choice(pool)))

        if self.style_cache is not None:
            # gather precomputed frozen style features by index -> (5, feat)
            s_imgs = self.style_cache[random_samples]
        else:
            s_imgs = torch.stack([self.transforms(self._img(i)) for i in random_samples])

        return img, transcr, wid, s_imgs, img_path, cor_im

    def _getitem_style(self, index):
        img = self._img(index)
        transcr = self.data[index][1]
        wid = self.data[index][2]

        pool = self._same_writer_pool(wid)
        positive = self._img(random.choice(pool))

        neg_wid = random.choice(self.writer_ids)
        while neg_wid == wid and self.wclasses > 1:
            neg_wid = random.choice(self.writer_ids)
        negative = self._img(random.choice(self.writer_to_indices[neg_wid]))

        samples = _sample(pool, 5)
        if self.transforms is not None:
            img = self.transforms(img)
            positive = self.transforms(positive)
            negative = self.transforms(negative)
            s_imgs = torch.stack([self.transforms(self._img(i)) for i in samples])
        else:
            s_imgs = None

        return img, transcr, wid, positive, negative, s_imgs
