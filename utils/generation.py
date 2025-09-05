import torch
import os
from PIL import Image
import torchvision
import cv2
import numpy as np


def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "images"), exist_ok=True)


def save_image_grid(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
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


def add_rescale_padding(
    words, fakes, max_word_length_width, longest_word_length, max_height=64
):
    # find the average character width of the max length word
    avg_char_width = max_word_length_width / longest_word_length
    scaled_padded_words = []

    for word, img in zip(words, fakes):
        img_pil = Image.fromarray(img)
        as_ratio = img_pil.width / img_pil.height
        # scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
        scaled_width = max(5, int(avg_char_width * len(word)))
        scaled_height = max(5, int(scaled_width / as_ratio))

        scaled_img = img_pil.resize((scaled_width, scaled_height))
        # print(f"Word {word} - scaled_img {scaled_img.size}")
        if word in PUNCTUATION:
            # rescale to height 10
            w_punc = scaled_img.width
            h_punc = scaled_img.height
            as_ratio_punct = w_punc / h_punc
            if word == ".":
                scaled_img = scaled_img.resize((int(5 * as_ratio_punct), 5))
            else:
                scaled_img = scaled_img.resize((int(13 * as_ratio_punct), 13))
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

                scaled_img = scaled_img.resize(
                    (int(max_height * as_ratio) - 4, max_height - 4)
                )
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


#####
# using the model
#####


def build_fake_image(
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
    im = crop_whitespace_width(im)
    im = Image.fromarray(im)
    im = np.array(im)
    return im
