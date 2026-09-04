import torch
import os
import math
from PIL import Image
import torchvision
import cv2
import numpy as np

OUTPUT_MAX_LEN = 95  # + 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64
PUNCTUATION = "_!\"#&'()*+,-./:;?"
INK_WIDTH_TOL = 1.15  # exact-placement crop may exceed real ink width by this factor


def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "images"), exist_ok=True)


def save_image_grid(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent:
        im = torchvision.transforms.ToPILImage()(grid)
        if not args.color:
            im = im.convert("L")
        else:
            im = im.convert("RGB")
    else:
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im


def _ink_bbox(img):
    """Tight (x, y, w, h) bbox of the ink in a grayscale image via Otsu, or
    ``None`` when the crop is blank (no foreground pixels)."""
    img_gray = np.array(img.convert("L"))
    ret, thresholded = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = cv2.findNonZero(thresholded)
    if coords is None:
        return None
    return cv2.boundingRect(coords)


def crop_whitespace_width(img):
    # tensor image to PIL
    bbox = _ink_bbox(img)
    if bbox is None:
        return np.array(img)
    x, y, w, h = bbox
    # rect = img.crop((x, 0, x + w, original_height))
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)


def split_long_word(word, max_chars=12):
    """Split a long word into non-overlapping chunks of at most ``max_chars``
    characters each, distributed as evenly as possible. Short words (and any
    non-positive ``max_chars``) are returned unchanged as a single chunk.

    Each chunk is generated on its own fixed 256-px canvas and later re-joined
    (see ``join_word_chunks``), so per-character resolution stays high and the
    hardcoded CANINE ``max_length`` truncation stops mattering.
    """
    if not max_chars or max_chars <= 0 or len(word) <= max_chars:
        return [word]
    nchunks = math.ceil(len(word) / max_chars)
    base, rem = divmod(len(word), nchunks)
    chunks = []
    i = 0
    for k in range(nchunks):
        size = base + (1 if k < rem else 0)
        chunks.append(word[i : i + size])
        i += size
    return chunks


def join_word_chunks(chunk_imgs, gap_width=4):
    """Horizontally butt-join the grayscale crops of a single word's chunks into
    one wide crop. Chunks are vertically centered onto a common (max) height and
    separated by a small white gap. Returns a PIL 'L' image."""
    if len(chunk_imgs) == 1:
        return chunk_imgs[0]
    arrs = [np.array(im.convert("L")) for im in chunk_imgs]
    max_h = max(a.shape[0] for a in arrs)
    gap = np.ones((max_h, gap_width), dtype=np.uint8) * 255
    pieces = []
    for idx, a in enumerate(arrs):
        pad_total = max_h - a.shape[0]
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        a = np.pad(
            a,
            ((pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        if idx > 0:
            pieces.append(gap)
        pieces.append(a)
    joined = np.concatenate(pieces, axis=1)
    return Image.fromarray(joined).convert("L")


def add_rescale_padding(
    words, fakes, max_word_length_width, longest_word_length, max_height=64
):
    # find the average character width of the max length word
    avg_char_width = max_word_length_width / longest_word_length
    scaled_padded_words = []

    for word, img in zip(words, fakes):
        img_pil = img
        as_ratio = img_pil.width / img_pil.height
        # scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
        scaled_width = max(5, int(avg_char_width * len(word)))
        scaled_height = max(5, int(scaled_width / as_ratio))

        scaled_img = img_pil.resize((scaled_width, scaled_height), Image.LANCZOS)
        # print(f"Word {word} - scaled_img {scaled_img.size}")
        if word in PUNCTUATION:
            # rescale to height 10
            w_punc = scaled_img.width
            h_punc = scaled_img.height
            as_ratio_punct = w_punc / h_punc
            if word == ".":
                scaled_img = scaled_img.resize(
                    (max(int(5 * as_ratio_punct), 5), 5), Image.LANCZOS
                )
            else:
                scaled_img = scaled_img.resize(
                    (max(int(13 * as_ratio_punct), 13), 13), Image.LANCZOS
                )
            # pad on top and leave the image in the bottom
            padding_bottom = 10
            padding_top = (
                max_height - scaled_img.height - padding_bottom
            )  # All padding goes on top
            # No padding at the bottom

            # Apply padding
            padded_img = np.pad(
                scaled_img,
                ((padding_top, padding_bottom), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        else:
            if scaled_img.height < max_height:
                padding = (max_height - scaled_img.height) // 2
                padded_img = np.pad(
                    scaled_img,
                    (
                        (padding, max_height - scaled_img.height - padding),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=255,
                )
            else:
                # resize to max height while maintaining aspect ratio
                # ar = scaled_img.width / scaled_img.height
                rsz_width = int(max_height * as_ratio) - 4
                rsz_height = max_height - 4

                rsz_width = max(3, rsz_width)
                rsz_height = max(3, rsz_height)

                scaled_img = scaled_img.resize((rsz_width, rsz_height), Image.LANCZOS)
                padding = (max_height - scaled_img.height) // 2
                padded_img = np.pad(
                    scaled_img,
                    (
                        (padding, max_height - scaled_img.height - padding),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=255,
                )

        scaled_padded_words.append(padded_img)
    return scaled_padded_words


def build_paragraph_image(
    scaled_padded_words, max_line_width=900, gap_height=64, gap_width=16
):
    gap = np.ones((gap_height, gap_width), dtype=np.uint8) * 255  # White gap
    current_line_width = 0
    # Concatenate images with gaps
    sentence_img = gap  # Start with a gap
    lines = []
    line_img = gap

    for img in scaled_padded_words:
        img_width = img.shape[1] + gap.shape[1]

        if current_line_width + img_width < max_line_width:
            # Add the image to the current line
            if line_img.shape[0] == 0:
                line_img = (
                    np.ones((gap_height, 0), dtype=np.uint8) * 255
                )  # Start a new line
            line_img = np.concatenate((line_img, img, gap), axis=1)
            current_line_width += img_width  # + gap.shape[1]
            # Check if adding this image exceeds the max line width
        else:
            # Pad the current line with white space to max_line_width
            remaining_width = max_line_width - current_line_width
            line_img = np.concatenate(
                (
                    line_img,
                    np.ones((gap_height, remaining_width), dtype=np.uint8) * 255,
                ),
                axis=1,
            )
            lines.append(line_img)

            # Start a new line with the current word
            line_img = np.concatenate((gap, img, gap), axis=1)
            current_line_width = img_width  # + 2 * gap.shape[1]
    # Add the last line to the lines list
    if current_line_width > 0:
        # Pad the last line to max_line_width
        remaining_width = max_line_width - current_line_width
        line_img = np.concatenate(
            (
                line_img,
                np.ones((gap_height, remaining_width), dtype=np.uint8) * 255,
            ),
            axis=1,
        )
        lines.append(line_img)

    paragraph_img_raw = np.concatenate((lines), axis=0)
    paragraph_image = Image.fromarray(paragraph_img_raw)
    paragraph_image = paragraph_image.convert("L")
    return paragraph_image


def upsample_word(im, upsampler=None, args=None, scale=2):
    """Return an upscaled (``scale``x) version of a single grayscale word crop.

    Uses the learned ``WordUpsampler`` when provided, otherwise falls back to a
    high-quality Lanczos resize. The output is uniformly ``scale``x larger in
    both dimensions, so aspect ratio (and thus downstream geometry) is preserved.
    """
    if upsampler is None:
        return im.resize((im.width * scale, im.height * scale), Image.LANCZOS)

    device = args.device
    x = torch.from_numpy(np.asarray(im.convert("L"), dtype=np.float32) / 255.0)
    x = x.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    upsampler.eval()
    with torch.no_grad():
        y = upsampler(x)
    y = y.clamp(0.0, 1.0).squeeze(0).squeeze(0).cpu().numpy()
    y = (y * 255.0).round().astype(np.uint8)
    return Image.fromarray(y).convert("L")


def upsample_words(fakes, args, models, scale=2):
    """Apply upsample_word to a list of word crops."""
    upsampler = models["upsampler"]
    return [upsample_word(im, upsampler, args, scale) for im in fakes]


def _median(values, fallback):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return float(fallback)
    return float(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))


def place_words_learned(
    words,
    fakes,
    writer_id,
    args,
    models,
    max_line_width=900,
    ref_height=64,
    left_margin=16,
    top_margin=16,
    seed=None,
    gap_clamp=(0.0, 5.0),
    base_clamp=(-1.0, 1.0),
    advance_jitter=0.05,
):
    """Lay the generated crops out with the stage-2 WordPlacer.

    The model predicts, per adjacent pair, a Gaussian over the horizontal gap
    and vertical baseline drift in units of ``H`` (see utils/placer_seq.py). Here
    ``H`` is recovered the same way as in training -- the **median ink-height of
    the generated crops** -- so the normalization matches. Predictions are
    sampled (clamped so words never overlap), then integrated left-to-right by a
    **deterministic greedy fill**: a word starts a new line whenever it would
    overflow ``max_line_width``; new lines drop by the writer's ``line_advance``
    (also in ``H`` units, carried on the model) plus small jitter. Pass ``seed``
    for reproducible sampling; different seeds give slight variation.
    """
    # imported lazily so the heavy torch text-encoder path is only pulled in when
    # learned placement is actually requested
    from utils.placer_seq import sequence_text_features

    placer = models["placer"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    diffusion = models["diffusion"]
    core = getattr(placer, "module", placer)

    n = len(fakes)
    if n == 0:
        return Image.new("L", (max_line_width, ref_height), color=255)

    # Shared reference scale, computed identically to training (median word height).
    H = _median([im.height for im in fakes], ref_height)

    device = args.device
    text_feats, _lengths = sequence_text_features(
        [list(words)], tokenizer, text_encoder, device
    )
    ink = torch.zeros((1, n, 2), dtype=torch.float32, device=device)
    after_punct = torch.zeros((1, n), dtype=torch.float32, device=device)
    for i, im in enumerate(fakes):
        ink[0, i, 0] = im.width / H
        ink[0, i, 1] = im.height / H
        if i > 0 and str(words[i - 1]) and str(words[i - 1])[-1] in PUNCTUATION:
            after_punct[0, i] = 1.0

    # Writer conditioning = the frozen style-bank vector (style-only placer), the
    # same [W, style_dim] tensor the diffusion UNet consumes. A writer with no
    # bank row falls back to zeros -> style_proj bias -> default spacing.
    style_dim = core.style_proj[0].in_features
    bank = getattr(diffusion, "style_bank", None)
    if bank is not None and 0 <= writer_id < bank.shape[0]:
        style_vec = bank[writer_id].to(device).unsqueeze(0)
    else:
        style_vec = torch.zeros((1, style_dim), device=device)

    placer.eval()
    with torch.no_grad():
        mu_gap, logvar_gap, mu_base, logvar_base = placer(
            text_feats, style_vec, ink, after_punct=after_punct
        )

    # Sample gap/base ~ N(mu, exp(logvar)) on CPU with a seedable generator.
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    def sample(mu, logvar, lo, hi):
        mu = mu[0].detach().cpu()
        std = torch.exp(0.5 * logvar[0].detach().cpu())
        eps = torch.randn(mu.shape, generator=gen)
        return torch.clamp(mu + std * eps, lo, hi)

    gap = sample(mu_gap, logvar_gap, *gap_clamp)
    base = sample(mu_base, logvar_base, *base_clamp)

    advance = core.line_advance
    n_writers = advance.shape[0]
    wa = float(advance[writer_id]) if 0 <= writer_id < n_writers else 4.0
    adv_eps = torch.randn(n, generator=gen)  # per-line jitter draws (indexed by word)

    placed = []  # (x, y, PIL image)
    cursor_x = left_margin
    line_center = top_margin + 0.5 * H
    prev_center = line_center
    for i, im in enumerate(fakes):
        gap_px = float(gap[i]) * H
        newline = (i == 0) or (cursor_x + gap_px + im.width > max_line_width)

        if newline:
            if i > 0:
                jitter = 1.0 + advance_jitter * float(adv_eps[i])
                line_center = line_center + max(wa * jitter, 1.2) * H
            cursor_x = left_margin
            cur_center = line_center
        else:
            cursor_x += gap_px
            cur_center = prev_center + float(base[i]) * H

        word_top = int(round(cur_center - 0.5 * im.height))
        placed.append((int(round(cursor_x)), max(0, word_top), im))
        cursor_x += im.width
        prev_center = cur_center

    canvas_w = max_line_width
    canvas_h = max((y + im.height for (x, y, im) in placed), default=ref_height)
    canvas_h += top_margin
    canvas = Image.new("L", (canvas_w, canvas_h), color=255)
    for x, y, im in placed:
        canvas.paste(im, (x, y))
    return canvas


def stack_images(images, margin=0, background="white"):
    """Stack PIL images vertically with optional uniform margin between and around them."""
    res_width = max(img.width for img in images) + 2 * margin
    res_height = sum(img.height for img in images) + margin * (len(images) + 1)
    dst = Image.new("RGB", (res_width, res_height), color=background)
    ch = margin
    for img in images:
        dst.paste(img, (margin, ch))
        ch += img.height + margin
    return dst


def _blank_patch(ref_gray, win=64):
    """Locate the brightest, most uniform ``win``x``win`` window in the page --
    i.e. blank paper. Returns the patch as a float32 array, or ``None`` when the
    page is too small."""
    arr = np.asarray(ref_gray.convert("L"), dtype=np.float32)
    ph, pw = arr.shape
    win = min(win, ph, pw)
    if win < 8:
        return None
    step = max(win // 2, 1)
    best = None
    for yy in range(0, ph - win + 1, step):
        for xx in range(0, pw - win + 1, step):
            tile = arr[yy : yy + win, xx : xx + win]
            # bright + low-structure scores highest (real ink lowers the mean and
            # raises the std, so text regions never win)
            score = tile.mean() - 2.0 * tile.std()
            if best is None or score > best[0]:
                best = (score, tile)
    return best[1] if best is not None else None


def paper_background(size, ref_gray, patch=64):
    """A paper-like background matching the reference page's tone and grain.

    Samples the brightest blank patch of ``ref_gray`` and stretches it to
    ``size`` for a seam-free low-frequency tone field, then adds procedural
    grain at the patch's own noise amplitude. Falls back to a flat page-tone
    fill + light grain when no page patch is available. ``size`` is (w, h);
    returns an 'L' image. A pure-white canvas is the giveaway this removes --
    scanned paper is never flat 255.
    """
    w, h = size
    tile = _blank_patch(ref_gray, patch)
    if tile is None:
        arr = np.asarray(ref_gray.convert("L"), dtype=np.float32)
        tone = float(arr.max()) if arr.size else 255.0
        field = np.full((h, w), tone, dtype=np.float32)
        grain = 2.0
    else:
        grain = float(tile.std())
        field = np.asarray(
            Image.fromarray(tile.astype(np.uint8)).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        )
    noise = np.random.normal(0.0, max(grain, 1.0), size=(h, w))
    bg = np.clip(field + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(bg, "L")


def _darken_paste(dst, crop, pos):
    """Composite a grayscale ``crop`` onto ``dst`` ('L') at ``pos`` by darkening
    (per-pixel min), so the ink stamps over the background but the crop's own
    white pixels let the paper below show through -- unlike an opaque paste,
    which would print a white box around every word."""
    x, y = pos
    cw, ch = crop.size
    dw, dh = dst.size
    if x >= dw or y >= dh:
        return
    cw = min(cw, dw - x)
    ch = min(ch, dh - y)
    if cw <= 0 or ch <= 0:
        return
    box = (x, y, x + cw, y + ch)
    base = np.asarray(dst.crop(box).convert("L"))
    top = np.asarray(crop.convert("L"))[:ch, :cw]
    blended = np.minimum(base, top)
    dst.paste(Image.fromarray(blended, "L"), (x, y))


def page_ink_level(ref_gray):
    """Representative ink darkness of a page: a low percentile of the sub-Otsu
    (dark) pixels. Not the absolute minimum (that's speckle) -- the 40th
    percentile of the ink pixels is a typical stroke value. Returns 0 when the
    page has no ink."""
    arr = np.asarray(ref_gray.convert("L"), dtype=np.uint8)
    thr, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = arr[arr < thr]
    if dark.size == 0:
        return 0.0
    return float(np.percentile(dark, 40))


def match_ink(crop, ref_ink, blur_sigma=0.6):
    """Darken a generated crop's ink to the page's ink level and soften it.

    The model's ink runs lighter and sharper than scanned ink. Remap the crop's
    own [ink, paper] range so its darkest strokes land at ``ref_ink`` while paper
    stays white (255) -- keeping the background white matters, so the later
    darken-composite leaves the paper between strokes untouched. A small Gaussian
    blur then matches the scanner's stroke softness. ``blur_sigma=0`` disables it.
    """
    arr = np.asarray(crop.convert("L"), dtype=np.float32)
    c_ink = float(np.percentile(arr, 5))
    c_paper = float(np.percentile(arr, 95))
    if c_paper - c_ink >= 1.0:
        scaled = np.clip((arr - c_ink) / (c_paper - c_ink), 0.0, 1.0)
        arr = ref_ink + scaled * (255.0 - ref_ink)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if blur_sigma and blur_sigma > 0:
        arr = cv2.GaussianBlur(arr, (0, 0), blur_sigma)
    return Image.fromarray(arr, "L")


def _paste_ink_matched(dst, fake, word, ref_gray, ref_ink=None):
    """Composite a generated word crop onto ``dst`` ('L') at ``word``'s real page
    position, darkening so the background shows between strokes.

    Sizes and positions the crop to the ORIGINAL ink measured in ``ref_gray``
    (the unmodified page, grayscale) inside the word's bbox, NOT the raw
    annotation bbox -- the bbox (esp. CSAFE's VOC boxes) runs looser than the
    ink, so scaling the tightly-cropped fake to fill it renders the word
    oversized and floated high. Falls back to the bbox when the box has no
    measurable ink. When ``ref_ink`` is given, the crop's ink is matched to that
    page ink level (see ``match_ink``) before compositing.
    """
    box = (word.x_start, word.y_start, word.x_end, word.y_end)
    ink = _ink_bbox(ref_gray.crop(box))
    if ink is None:
        tx, ty, target_h = word.x_start, word.y_start, word.height
        target_w = word.x_end - word.x_start
    else:
        ix, iy, iw, ih = ink
        tx, ty, target_h, target_w = word.x_start + ix, word.y_start + iy, ih, iw
    ratio = target_h / max(fake.height, 1)
    scaled_width = max(int(fake.width * ratio), 3)
    scaled_height = max(target_h, 3)
    # clamp a crop wider than the real ink (+tol) so it can't sprawl over
    # neighbours; shrink height with it to keep the glyph's aspect ratio
    max_width = max(int(target_w * INK_WIDTH_TOL), 3)
    if scaled_width > max_width:
        scaled_height = max(int(scaled_height * max_width / scaled_width), 3)
        scaled_width = max_width
    scaled_img = fake.resize((scaled_width, scaled_height), Image.LANCZOS)
    if ref_ink is not None:
        scaled_img = match_ink(scaled_img, ref_ink)
    _darken_paste(dst, scaled_img, (tx, ty))


def build_ref_paragraph(fakes, xpr, raw_orig):
    """Regenerate a whole paragraph in place: every word replaced by its
    generated crop at the word's real page position, on a blank canvas.

    ``fakes`` is aligned 1:1 with ``xpr.words``. Each crop is sized/placed to the
    original ink measured on ``raw_orig`` (see ``_paste_ink_matched``), so the
    exact-placement dupe matches the real page's geometry instead of the looser
    annotation bboxes. Returns the paragraph region as a grayscale image.
    """
    assert len(xpr.words) == len(fakes)
    ref_gray = raw_orig.convert("L")
    ref_ink = page_ink_level(ref_gray)
    dupe = paper_background((xpr.img_width, xpr.img_height), ref_gray)
    for fake, word in zip(fakes, xpr.words):
        _paste_ink_matched(dupe, fake, word, ref_gray, ref_ink)
    return xpr.get_cropped(dupe)


def build_replaced_paragraph(raw_orig, xpr, gen_crops, replace_indices):
    """Composite generated word crops over selected bboxes of a real IAM form.

    Starts from the real form image (real ink everywhere) and, for each index in
    ``replace_indices``, clears that word's XML bbox to matched paper (not flat
    white, which would leave a clean rectangle on textured paper) and darkens
    the matching generated crop over it, sized/placed to the original ink
    measured inside that bbox (see ``_paste_ink_matched``). Every other word
    keeps its original real ink. Returns the paragraph region as a grayscale
    image.

    ``gen_crops`` is aligned 1:1 with ``replace_indices``.
    """
    assert len(gen_crops) == len(replace_indices)
    ref_gray = raw_orig.convert("L")
    ref_ink = page_ink_level(ref_gray)
    dupe = raw_orig.convert("L")
    for fake, i in zip(gen_crops, replace_indices):
        word = xpr.words[i]
        box = (word.x_start, word.y_start, word.x_end, word.y_end)
        # erase the original ink with paper matched to the page, then darken the
        # generated crop over it so the cleared region carries the page's grain
        patch = paper_background((word.x_end - word.x_start, word.y_end - word.y_start), ref_gray)
        dupe.paste(patch, box)
        _paste_ink_matched(dupe, fake, word, ref_gray, ref_ink)
    return xpr.get_cropped(dupe)


def capture_png(img, out_path, noise_sigma=3.0):
    """Save ``img`` through one uniform 'capture' pass, as PNG.

    Meant to be applied to BOTH the real reference crop and every dupe so their
    low-level statistics -- and file size -- share one distribution: convert to
    grayscale, add a faint common sensor-noise floor, and re-encode as PNG.
    Because a scanned page carries grain everywhere while a synthetic composite
    starts pristine, a shared noise floor + identical PNG encoding is what pulls
    their byte sizes together; running it symmetrically keeps the real-vs-fake
    comparison fair. ``noise_sigma=0`` still re-encodes (format/mode parity).
    """
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    if noise_sigma and noise_sigma > 0:
        arr = arr + np.random.normal(0.0, noise_sigma, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, "L").save(out_path, format="PNG")


def compose_on_paper(ink_img, ref_gray):
    """Drop an ink-on-white grayscale paragraph (the heuristic-reflow builders'
    output) onto paper matched to ``ref_gray``, darkening so only the ink prints.
    Removes the flat-white background that makes those variants trivial to spot,
    and matches the ink to the page's ink level (see ``match_ink``)."""
    ink = match_ink(ink_img, page_ink_level(ref_gray))
    bg = paper_background(ink.size, ref_gray)
    _darken_paste(bg, ink, (0, 0))
    return bg


#####
# using the model
#####


def _sampling_models(models):
    """Unpack the load_models() bundle into the ordered tuple the sampling
    builders use: (diffusion, ema_model, vae, feature_extractor, ddim,
    transform, tokenizer, text_encoder)."""
    return (
        models["diffusion"],
        models["ema_model"],
        models["vae"],
        models["feature_extractor"],
        models["ddim"],
        models["transform"],
        models["tokenizer"],
        models["text_encoder"],
    )


def build_fake_image_1(
    word,
    writer_id,
    args,
    models,
    crop_whitespace=True,
):
    (diffusion, ema_model, vae, feature_extractor, ddim, transform,
     tokenizer, text_encoder) = _sampling_models(models)
    # print("Word:", word)
    labels = torch.tensor([writer_id]).long().to(args.device)
    ema_sampled_images = diffusion.sampling(
        ema_model,
        vae,
        n=len(labels),
        x_text=word,
        labels=labels,
        args=args,
        style_extractor=feature_extractor,
        noise_scheduler=ddim,
        transform=transform,
        character_classes=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        run_idx=None,
    )
    image = ema_sampled_images.squeeze(0)
    im = torchvision.transforms.ToPILImage()(image)
    im = im.convert("L")
    if crop_whitespace:
        im = crop_whitespace_width(im)
        im = Image.fromarray(im)
    return im


def build_fake_interp_1(
    args,
    models,
    crop_whitespace=True,
):
    (diffusion, ema_model, vae, feature_extractor, ddim, transform,
     tokenizer, text_encoder) = _sampling_models(models)
    # print("Word:", word)
    word = args.sampling_word
    writer_1 = args.writer_1
    writer_2 = args.writer_2
    labels = torch.tensor([writer_1, writer_2]).long().to(args.device)
    ema_sampled_images = diffusion.interp_1(
        ema_model,
        vae,
        x_text=word,
        labels=labels,
        args=args,
        style_extractor=feature_extractor,
        noise_scheduler=ddim,
        transform=transform,
        character_classes=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        run_idx=None,
    )
    image = ema_sampled_images.squeeze(0)
    im = torchvision.transforms.ToPILImage()(image)
    im = im.convert("L")
    if crop_whitespace:
        im = crop_whitespace_width(im)
        im = Image.fromarray(im)
    return im


def build_fake_image_N(
    words,
    s,
    args,
    models,
    longest_word_length,
    max_word_length_width,
    crop_whitespace=True,
):
    (diffusion, ema_model, vae, feature_extractor, ddim, transform,
     tokenizer, text_encoder) = _sampling_models(models)
    labels = torch.tensor([s]).long().to(args.device)

    # Expand long words into fixed-canvas chunks; each chunk is generated as its
    # own "word" in the same batched pass (same writer style), then the chunks of
    # a word are re-joined into a single wide crop so callers still see one crop
    # per original word.
    max_word_chars = getattr(args, "max_word_chars", 0)
    pieces = []
    owners = []
    for wi, word in enumerate(words):
        for chunk in split_long_word(word, max_word_chars):
            pieces.append(chunk)
            owners.append(wi)

    ema_sampled_images = diffusion.sampling_bulk(
        ema_model,
        vae,
        x_text=pieces,
        labels=labels,
        args=args,
        style_extractor=feature_extractor,
        noise_scheduler=ddim,
        transform=transform,
        character_classes=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        run_idx=None,
    )
    topil = torchvision.transforms.ToPILImage()
    word_chunks = [[] for _ in words]
    for j in range(len(pieces)):
        image = ema_sampled_images[j].squeeze(0)
        im = topil(image)
        im = im.convert("L")
        if crop_whitespace:
            im = crop_whitespace_width(im)
            im = Image.fromarray(im)
        word_chunks[owners[j]].append(im)

    fakes = []
    for wi, word in enumerate(words):
        im = join_word_chunks(word_chunks[wi])
        if len(word) == longest_word_length:
            max_word_length_width = im.width
        fakes.append(im)
    return fakes, max_word_length_width


def build_fake_interp_N(
    words,
    args,
    models,
    longest_word_length,
    max_word_length_width,
    crop_whitespace=True,
):
    (diffusion, ema_model, vae, feature_extractor, ddim, transform,
     tokenizer, text_encoder) = _sampling_models(models)
    fakes = []
    writer_1 = args.writer_1
    writer_2 = args.writer_2
    labels = torch.tensor([writer_1, writer_2]).long().to(args.device)
    ema_sampled_images = diffusion.interp_bulk(
        ema_model,
        vae,
        x_text=words,
        labels=labels,
        args=args,
        style_extractor=feature_extractor,
        noise_scheduler=ddim,
        transform=transform,
        character_classes=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        run_idx=None,
    )
    topil = torchvision.transforms.ToPILImage()
    for i in range(len(words)):
        word = words[i]
        image = ema_sampled_images[i].squeeze(0)
        im = topil(image)
        im = im.convert("L")
        if crop_whitespace:
            im = crop_whitespace_width(im)
            im = Image.fromarray(im)
        if len(word) == longest_word_length:
            max_word_length_width = im.width
        fakes.append(im)
    return fakes, max_word_length_width


def render_paragraph(words, args, models, max_line_width, s=None, interp=False):
    """Generate crops for ``words`` and lay them out as one paragraph image.

    Bundles the three-call sequence repeated across the generation scripts:
    ``build_fake_{image,interp}_N`` -> ``add_rescale_padding`` ->
    ``build_paragraph_image``. Static-style paragraphs pass ``s`` (writer id);
    interpolated paragraphs pass ``interp=True`` and set
    ``args.writer_1``/``writer_2``/``mix_rate`` (the interp builder reads the
    writers off ``args``).
    """
    longest_word_length = max(len(word) for word in words)
    if interp:
        fakes, max_word_length_width = build_fake_interp_N(
            words,
            args,
            models,
            longest_word_length=longest_word_length,
            max_word_length_width=0,
        )
    else:
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            args,
            models,
            longest_word_length=longest_word_length,
            max_word_length_width=0,
        )
    scaled_padded_words = add_rescale_padding(
        words,
        fakes,
        max_word_length_width=max_word_length_width,
        longest_word_length=longest_word_length,
    )
    return build_paragraph_image(scaled_padded_words, max_line_width=max_line_width)
