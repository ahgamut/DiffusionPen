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


def crop_whitespace_width(img):
    # tensor image to PIL
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
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


#####
# using the model
#####


def build_fake_image_1(
    word,
    writer_id,
    args,
    diffusion,
    ema_model,
    vae,
    feature_extractor,
    ddim,
    transform,
    tokenizer,
    text_encoder,
    crop_whitespace=True,
):
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
    diffusion,
    ema_model,
    vae,
    feature_extractor,
    ddim,
    transform,
    tokenizer,
    text_encoder,
    crop_whitespace=True,
):
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
    diffusion,
    ema_model,
    vae,
    feature_extractor,
    ddim,
    transform,
    tokenizer,
    text_encoder,
    longest_word_length,
    max_word_length_width,
    crop_whitespace=True,
):
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
    diffusion,
    ema_model,
    vae,
    feature_extractor,
    ddim,
    transform,
    tokenizer,
    text_encoder,
    longest_word_length,
    max_word_length_width,
    crop_whitespace=True,
):
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
