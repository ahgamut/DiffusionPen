"""Synthetic "font writer" generation for the merged word dataset.

Each font file becomes one synthetic *writer*; its words are rendered by PIL and
augmented toward a handwritten look (mesh warp + affine slant + stroke-width and
grayscale jitter). The point is coverage: fonts render any glyph / letter-
transition -- including the ones a closed-vocabulary corpus (CSAFE) never shows --
with an exact, noise-free transcription.

This module is self-contained (PIL + numpy only, no ``build_multidataset`` import).
``font_records`` is the adapter the builder calls; it returns the same
``record = {dataset, writer, transcr, ref}`` dicts every other adapter yields, with
``ref = ("font", font_path, text, params)`` -- ``params`` holds the fully-resolved,
compact per-instance render/augment scalars so ``render_and_augment`` reproduces the
crop deterministically in the builder's write pass. Real-crop access (for color
stats) is injected as a ``crop_loader`` callable to avoid a circular import.

Nothing here runs in this env (no PIL/numpy); verified by import-trace only.
"""

import glob
import math
import os
import string
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# chars a font must mostly cover to be usable, and the render/geometry defaults.
_REQUIRED = string.ascii_lowercase
_ALLOWED_WORD = set(string.ascii_letters)
_CHECK_SIZE = 40  # px used only for the glyph-coverage probe


@dataclass
class FontSynthConfig:
    """Knobs for synthetic font generation (the builder maps CLI flags onto this)."""

    word_source: str = "wordfreq"      # "wordfreq" | "nltk" | path to a wordlist file
    candidate_limit: int = 5000        # cap on candidate words scored for coverage
    words_per_writer: int = 400        # distinct words each font writer renders
    instances_per_word: int = 4        # augmented renders per (writer, word)
    share_words: bool = True           # all writers render the same target set
    ngram_fill: bool = True            # bias word choice toward rare real-corpus n-grams
    ngram_sizes: tuple = (2, 3)        # n-gram sizes used for coverage scoring
    color_from_real: bool = True       # estimate ink/bg grayscale from real crops
    color_sample: int = 500            # #real crops sampled for color stats
    size_min: int = 48                 # font pixel-size range
    size_max: int = 64
    rotation_deg: float = 5.0          # affine bounds (per instance, uniform +/-)
    shear_deg: float = 12.0
    max_blur: float = 0.6              # gaussian-blur radius upper bound
    max_noise: float = 6.0             # additive-noise std upper bound (0..255)
    cap_title_p: float = 0.20          # per-instance chance to Title-case (uppercase coverage)
    cap_upper_p: float = 0.08          # per-instance chance to UPPER-case
    max_missing_lower: int = 2         # skip a font missing more than this many lowercase glyphs
    seed: int = 0

    # default grayscale model used when color_from_real is off / yields nothing.
    default_ink: tuple = (45.0, 15.0)  # (mean, std)
    default_bg: tuple = (245.0, 6.0)
    ink_jitter: float = 8.0            # per-instance grayscale jitter around writer mean
    bg_jitter: float = 4.0


# --------------------------------------------------------------------------- #
# word sourcing + rare-n-gram-targeted selection
# --------------------------------------------------------------------------- #
def load_candidate_words(source, limit):
    """Return up to ``limit`` letters-only candidate words from a pluggable source.

    ``source`` is ``"wordfreq"`` (frequency-ranked English), ``"nltk"``
    (``nltk.corpus.words``), or a path to a newline-delimited wordlist file.
    """
    if source == "wordfreq":
        from wordfreq import top_n_list

        raw = top_n_list("en", limit * 3)
    elif source == "nltk":
        from nltk.corpus import words as nltk_words

        raw = nltk_words.words()
    else:
        if not os.path.isfile(source):
            raise ValueError(f"--font-word-source {source!r} is not 'wordfreq'/'nltk' or a file")
        with open(source, encoding="utf-8") as f:
            raw = [ln.strip() for ln in f]

    out, seen = [], set()
    for w in raw:
        w = w.strip()
        lw = w.lower()
        if lw in seen or not (2 <= len(w) <= 15):
            continue
        if any(c not in _ALLOWED_WORD for c in w):
            continue
        seen.add(lw)
        out.append(lw)
        if len(out) >= limit:
            break
    return out


def _ngrams(word, sizes):
    s = set()
    for n in sizes:
        for i in range(len(word) - n + 1):
            s.add(word[i : i + n])
    return s


def real_ngram_counts(records, sizes=(1, 2, 3)):
    """Char n-gram counts over the real transcriptions -- to know what's already common."""
    c = Counter()
    for r in records:
        t = r["transcr"]
        for n in sizes:
            for i in range(len(t) - n + 1):
                c[t[i : i + n]] += 1
    return c


def select_target_words(candidates, real_counts, n_words, sizes=(2, 3), fill=True):
    """Greedy set-cover: pick words that maximally add *rare* real-corpus n-grams.

    Rarity weight of an n-gram = 1/(1+real_count); a running tally gives diminishing
    returns so the picked set spreads coverage instead of piling onto one gap.
    ``fill=False`` just takes the first ``n_words`` (already frequency-ordered).
    """
    if not fill:
        return list(candidates)[:n_words]

    cand = [[w, _ngrams(w, sizes)] for w in candidates]
    covered = Counter()
    chosen = []
    k = min(n_words, len(cand))
    for _ in range(k):
        best_i, best_score = -1, -1.0
        for i, entry in enumerate(cand):
            w, ngs = entry
            if w is None:
                continue
            score = 0.0
            for ng in ngs:
                score += (1.0 / (1.0 + real_counts.get(ng, 0))) / (1.0 + covered[ng])
            if score > best_score:
                best_score, best_i = score, i
        if best_i < 0:
            break
        w, ngs = cand[best_i]
        chosen.append(w)
        for ng in ngs:
            covered[ng] += 1
        cand[best_i][0] = None
    return chosen


# --------------------------------------------------------------------------- #
# color model (match synthetic ink/paper to the real data)
# --------------------------------------------------------------------------- #
def estimate_color_stats(real_records, crop_loader, sample=500, seed=0):
    """Grayscale (mean,std) of dark ink vs light background over a real-crop sample.

    ``crop_loader(record) -> PIL.Image | None`` is injected by the builder so this
    module needs no knowledge of the ref grammar / page cache.
    """
    if not real_records:
        return None
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(real_records), size=min(sample, len(real_records)), replace=False)
    ink, bg = [], []
    for i in idx:
        img = crop_loader(real_records[int(i)])
        if img is None:
            continue
        g = np.asarray(img.convert("L")).ravel()
        ink.append(g[g < 100])
        bg.append(g[g > 200])
    ink = np.concatenate(ink) if ink else np.array([])
    bg = np.concatenate(bg) if bg else np.array([])
    if ink.size < 100 or bg.size < 100:
        return None
    return {
        "ink": (float(ink.mean()), float(ink.std()) or 1.0),
        "bg": (float(bg.mean()), float(bg.std()) or 1.0),
    }


# --------------------------------------------------------------------------- #
# glyph-coverage probe (skip fonts/words that would render .notdef boxes)
# --------------------------------------------------------------------------- #
def _glyph_key(font, ch):
    m = font.getmask(ch, mode="L")
    return bytes(m) if m else b""


def font_missing_chars(font_path, chars, size=_CHECK_SIZE):
    """Chars in ``chars`` that map to the font's .notdef glyph (compare vs a PUA codepoint)."""
    font = ImageFont.truetype(font_path, size)
    notdef = _glyph_key(font, "")  # private-use codepoint fonts virtually never define
    missing = set()
    for ch in chars:
        if ch == " ":
            continue
        if _glyph_key(font, ch) == notdef:
            missing.add(ch)
    return missing


# --------------------------------------------------------------------------- #
# rendering + augmentation
# --------------------------------------------------------------------------- #
def render_word(text, font_path, size, ink_gray, bg_gray, margin=8):
    """Render ``text`` as a tight variable-width RGB crop (ink on paper grayscale)."""
    font = ImageFont.truetype(font_path, size)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    l, t, r, b = tmp.textbbox((0, 0), text, font=font)
    w, h = max(1, r - l), max(1, b - t)
    img = Image.new("RGB", (w + 2 * margin, h + 2 * margin), (bg_gray,) * 3)
    ImageDraw.Draw(img).text((margin - l, margin - t), text, font=font, fill=(ink_gray,) * 3)
    return img


def _mesh_warp(img, rng, fill, cells=(4, 4), max_disp=0.08):
    """PIL MESH warp: jitter interior grid vertices, keep the border fixed."""
    w, h = img.size
    gx, gy = cells
    xs = np.linspace(0, w, gx + 1)
    ys = np.linspace(0, h, gy + 1)
    dmx, dmy = max_disp * w / gx, max_disp * h / gy
    pt = {}
    for i in range(gx + 1):
        for j in range(gy + 1):
            dx = 0.0 if i in (0, gx) else rng.uniform(-dmx, dmx)
            dy = 0.0 if j in (0, gy) else rng.uniform(-dmy, dmy)
            pt[(i, j)] = (xs[i] + dx, ys[j] + dy)
    mesh = []
    for i in range(gx):
        for j in range(gy):
            dest = (int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1]))
            nw, sw, se, ne = pt[(i, j)], pt[(i, j + 1)], pt[(i + 1, j + 1)], pt[(i + 1, j)]
            quad = (nw[0], nw[1], sw[0], sw[1], se[0], se[1], ne[0], ne[1])
            mesh.append((dest, quad))
    return img.transform((w, h), Image.MESH, mesh, resample=Image.BILINEAR, fillcolor=fill)


def _affine_slant(img, rot_deg, shear_deg, fill):
    """Rotate then horizontally shear (slant), growing the canvas so nothing clips."""
    img = img.rotate(rot_deg, resample=Image.BILINEAR, expand=True, fillcolor=fill)
    sh = math.tan(math.radians(shear_deg))
    w, h = img.size
    neww = w + int(abs(sh) * h)
    dx = -min(0.0, sh) * h  # keep content in-frame for either slant direction
    return img.transform((neww, h), Image.AFFINE, (1, sh, dx, 0, 1, 0),
                         resample=Image.BILINEAR, fillcolor=fill)


def augment(img, params):
    """Apply the resolved per-instance augmentation to a rendered word crop."""
    fill = (params["bg"],) * 3
    rng = np.random.RandomState(params["warp_seed"])

    stroke = params["stroke"]
    if stroke > 0:
        img = img.filter(ImageFilter.MinFilter(3))   # thicken dark ink
    elif stroke < 0:
        img = img.filter(ImageFilter.MaxFilter(3))    # thin it

    img = _mesh_warp(img, rng, fill)
    img = _affine_slant(img, params["rot"], params["shear"], fill)

    if params["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(params["blur"]))
    if params["noise"] > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr += rng.normal(0.0, params["noise"], size=arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return img


def render_and_augment(font_path, text, params):
    """The builder's ``load_crop`` hook: reproduce one synthetic crop from its params."""
    img = render_word(text, font_path, params["size"], params["ink"], params["bg"])
    return augment(img, params)


# --------------------------------------------------------------------------- #
# the adapter
# --------------------------------------------------------------------------- #
def _writer_color_means(stats, cfg, rng):
    ink_m, ink_s = stats["ink"] if stats else cfg.default_ink
    bg_m, bg_s = stats["bg"] if stats else cfg.default_bg
    ink = float(np.clip(rng.normal(ink_m, ink_s), 0, 120))
    bg = float(np.clip(rng.normal(bg_m, bg_s), 160, 255))
    return ink, bg


def _resolve_params(cfg, rng, ink_mean, bg_mean):
    return {
        "size": int(rng.randint(cfg.size_min, cfg.size_max + 1)),
        "ink": int(np.clip(rng.normal(ink_mean, cfg.ink_jitter), 0, 130)),
        "bg": int(np.clip(rng.normal(bg_mean, cfg.bg_jitter), 150, 255)),
        "rot": float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg)),
        "shear": float(rng.uniform(-cfg.shear_deg, cfg.shear_deg)),
        "stroke": int(rng.choice([-1, 0, 0, 1])),
        "blur": float(rng.uniform(0.0, cfg.max_blur)),
        "noise": float(rng.uniform(0.0, cfg.max_noise)),
        "warp_seed": int(rng.randint(2**31 - 1)),
    }


def _case_word(word, cfg, rng):
    u = rng.rand()
    if u < cfg.cap_upper_p:
        return word.upper()
    if u < cfg.cap_upper_p + cfg.cap_title_p:
        return word.capitalize()
    return word


def font_records(font_dir, real_records, crop_loader, cfg):
    """Build synthetic font-writer records: one writer per font file.

    ``real_records`` (the other adapters' output) drives rare-n-gram word targeting
    and the ink/paper color model; ``crop_loader`` loads real crops for color stats.
    """
    fonts = sorted(
        glob.glob(os.path.join(font_dir, "*.ttf")) + glob.glob(os.path.join(font_dir, "*.otf"))
    )
    if not fonts:
        raise SystemExit(f"no .ttf/.otf fonts under {font_dir}")

    candidates = load_candidate_words(cfg.word_source, cfg.candidate_limit)
    real_counts = real_ngram_counts(real_records) if real_records else Counter()
    targets = select_target_words(
        candidates, real_counts, cfg.words_per_writer, cfg.ngram_sizes, fill=cfg.ngram_fill
    )

    stats = estimate_color_stats(real_records, crop_loader, cfg.color_sample, cfg.seed) \
        if cfg.color_from_real else None

    rng = np.random.RandomState(cfg.seed)
    recs = []
    for font_path in fonts:
        writer = os.path.splitext(os.path.basename(font_path))[0]
        missing = font_missing_chars(font_path, _REQUIRED)
        if len(missing) > cfg.max_missing_lower:
            print(f"  [font] skip {writer}: missing lowercase {sorted(missing)}")
            continue
        bad = font_missing_chars(font_path, string.ascii_uppercase + string.digits)

        ink_mean, bg_mean = _writer_color_means(stats, cfg, rng)
        if cfg.share_words:
            words = targets
        else:
            take = rng.choice(len(targets), size=min(cfg.words_per_writer, len(targets)), replace=False)
            words = [targets[int(i)] for i in take]

        n_writer = 0
        for word in words:
            for _ in range(cfg.instances_per_word):
                text = _case_word(word, cfg, rng)
                if any(c in bad for c in text):  # coverage guard: don't emit .notdef boxes
                    continue
                params = _resolve_params(cfg, rng, ink_mean, bg_mean)
                recs.append(
                    {
                        "dataset": "FONT",
                        "writer": writer,
                        "transcr": text,
                        "ref": ("font", font_path, text, params),
                    }
                )
                n_writer += 1
        print(f"  [font] {writer}: {n_writer} crops")
    return recs
