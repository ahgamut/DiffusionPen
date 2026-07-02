import argparse
import os
import json
import pandas as pd
from PIL import Image, ImageOps
import glob
import sys
import traceback
import random
import torch
import skimage.filters as skfilt

#
from models.diffpen2 import IAM_TempLoader
from utils.generation import (
    setup_logging,
    build_fake_image_N,
    build_fake_interp_N,
    add_rescale_padding,
    build_paragraph_image,
)
from utils.relcharsize import build_placed_paragraph
from utils.relcharsize import build_bbox_place_paragraph
from utils.relcharsize import get_possible_font_sizes
from utils.subprompt import Word, Prompt
from utils.arghandle import add_common_args
from utils.model_setup import load_models


class CTX:
    mldict = dict()


def save_threshed(img, fname):
    import numpy as np
    arr = np.array(img.convert("L"))
    thr = np.array(arr > skfilt.threshold_otsu(arr), dtype=np.uint8)
    thr = 255 * thr
    timg = Image.fromarray(thr).convert("L")
    timg.save(fname)


def resave_real(xmlname, imgname, targname):
    prompt = Prompt(xmlname)
    img = Image.open(imgname).convert("RGB")
    crop = prompt.get_cropped(img)
    save_threshed(crop, targname)


def build_ref_paragraph(fakes, xpr, max_line_width, longest_word_length):
    assert len(xpr.words) == len(fakes)
    dupe = Image.new("RGB", size=(xpr.img_width, xpr.img_height), color="white")

    for i in range(len(fakes)):
        word = xpr.words[i]
        fake = fakes[i]
        ratio = word.height / fake.height
        #
        scaled_width = int(fake.width * ratio)
        scaled_height = word.height
        scaled_width = max(scaled_width, 3)
        scaled_height = max(scaled_height, 3)
        scaled_img = fakes[i].resize((scaled_width, scaled_height))
        dupe.paste(scaled_img, (word.x_start, word.y_start))

    dupe = dupe.convert("L")
    return xpr.get_cropped(dupe)


def make_closedset(fname, targdir):
    df = pd.read_csv(fname)
    for ind, row in df.iterrows():
        wid = row["wid"].replace('"', "")
        xmlname = os.path.join("./iam_data/xml/", row["xmlname"])
        imgname = os.path.join("./iam_data/forms", row["imgname"])
        targname = os.path.join(targdir, row["target_name"]) + ".png"
        resave_real(xmlname, imgname, targname)


def resave_fake(xmlname, imgname, targname, faketype):
    xpr = Prompt(xmlname)
    raw_orig = Image.open(os.path.join("./iam_data", "forms", xpr.idd + ".png"))
    s = IAM_TempLoader.map_wid_to_index(xpr.writer_id)
    if "niceplace" in faketype:
        print("should regenerate", imgname, "place nicely and save")
        words = [w.raw for w in xpr.words]
        longest_word_length = max(len(word) for word in words)
        raw_crop = xpr.get_cropped(raw_orig)
        max_line_width = raw_crop.width
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
        regen_img = build_ref_paragraph(
            fakes,
            xpr,
            max_line_width=max_line_width,
            longest_word_length=longest_word_length,
        )
        save_threshed(regen_img, targname)
        return

    if "traintext" in faketype:
        print("should regenerate", imgname, "place however and save")
        words = [w.raw for w in xpr.words]
        longest_word_length = max(len(word) for word in words)
        max_line_width = CTX.mldict["args"].max_line_width
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
    elif "difftext1" in faketype:
        print("should generate LL using wid from", imgname, "and save")
        lines = open("./prompts/london-letter.txt").read()
        words = lines.strip().split(" ")
        longest_word_length = max(len(word) for word in words)
        max_line_width = CTX.mldict["args"].max_line_width
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
    elif "difftext2" in faketype:
        print("should generate WOZ using wid from", imgname, "and save")
        lines = open("./prompts/woz-letter.txt").read()
        words = lines.strip().split(" ")
        longest_word_length = max(len(word) for word in words)
        max_line_width = CTX.mldict["args"].max_line_width
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )

    postparts = faketype.split("-")
    if len(postparts) == 2:
        scaled_padded_words = add_rescale_padding(
            words,
            fakes,
            max_word_length_width=max_word_length_width,
            longest_word_length=longest_word_length,
        )
        regen_img = build_paragraph_image(
            scaled_padded_words, max_line_width=max_line_width
        )
    elif len(postparts) == 4:
        use_aspect = postparts[3] == "img"
        if postparts[2] == "rsp":
            possible_font_sizes = get_possible_font_sizes(xpr.words, dpi=600)
            regen_img = build_bbox_place_paragraph(
                words,
                fakes,
                possible_font_sizes,
                max_line_width=max_line_width,
                dpi=600,
                use_aspect=use_aspect,
            )
        else:
            font_size = int(postparts[2])
            regen_img = build_placed_paragraph(
                words,
                fakes,
                max_line_width=max_line_width,
                font_size=font_size,
                dpi=600,
                use_aspect=use_aspect,
            )
    else:
        raise RuntimeError("invalid post-processing:" + faketype)

    save_threshed(regen_img, targname)


def resave_interp(xmlname, imgname, targname, widinfo, interp):
    wid1, wid2, alpha = widinfo.split("-")
    wid1 = wid1.replace('"', "")
    wid2 = wid2.replace('"', "")
    alpha = float(alpha)
    if "sametext" in interp:
        print(
            "should interpolate between",
            (wid1, wid2),
            "at",
            alpha,
            "use same text and save to",
            targname,
        )
        xpr = Prompt(xmlname)
        words = [w.raw for w in xpr.words]
        longest_word_length = max(len(word) for word in words)
        raw_orig = Image.open(os.path.join("./iam_data", "forms", xpr.idd + ".png"))
        raw_crop = xpr.get_cropped(raw_orig)
        s1 = IAM_TempLoader.map_wid_to_index(wid1)
        s2 = IAM_TempLoader.map_wid_to_index(wid2)
        max_line_width = raw_crop.width
        CTX.mldict["args"].writer_1 = s1
        CTX.mldict["args"].writer_2 = s2
        CTX.mldict["args"].mix_rate = alpha
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_interp_N(
            words,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
        regen_img2 = build_ref_paragraph(
            fakes,
            xpr,
            max_line_width=max_line_width,
            longest_word_length=longest_word_length,
        )
        save_threshed(regen_img2, targname)
    else:
        print(
            "should interpolate between",
            (wid1, wid2),
            "at",
            alpha,
            "use different text and save to",
            targname,
        )
        xpr = Prompt(xmlname)
        lines = open("./prompts/london-letter.txt").read()
        words = lines.strip().split(" ")
        longest_word_length = max(len(word) for word in words)
        s1 = IAM_TempLoader.map_wid_to_index(wid1)
        s2 = IAM_TempLoader.map_wid_to_index(wid2)
        max_line_width = CTX.mldict["args"].max_line_width
        CTX.mldict["args"].writer_1 = s1
        CTX.mldict["args"].writer_2 = s2
        CTX.mldict["args"].mix_rate = alpha
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_interp_N(
            words,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
        regen_img = build_placed_paragraph(
            words,
            fakes,
            max_line_width=max_line_width,
            font_size=16,
            dpi=600,
            use_aspect=random.random() < 0.5,
        )
        save_threshed(regen_img, targname)


def process_csv(fname, targdir):
    print("processing", fname)
    if "clref" in fname:
        make_closedset(fname, targdir)
        return
    #
    df = pd.read_csv(fname)
    for ind, row in df.iterrows():
        wid = row["file2_wid"].replace('"', "")
        proc_tp = row["file2_type"]
        targname = os.path.join(targdir, row["target_name"]) + ".png"

        if proc_tp == "real":
            img_basename = row["file2_path"]
            imgname = os.path.join("./iam_data/forms", img_basename)
            xmlname = os.path.join("./iam_data/xml", img_basename.replace("png", "xml"))
            resave_real(xmlname, imgname, targname)

        elif proc_tp.startswith("fake-"):
            img_basename = row["file2_path"]
            imgname = os.path.join("./iam_data/forms", img_basename)
            xmlname = os.path.join("./iam_data/xml", img_basename.replace("png", "xml"))
            resave_fake(xmlname, imgname, targname, proc_tp)

        else:
            anchor_basename = row["file1_path"]
            imgname = os.path.join("./iam_data/forms", anchor_basename)
            xmlname = os.path.join(
                "./iam_data/xml", anchor_basename.replace("png", "xml")
            )
            widinfo = row["file2_path"]
            resave_interp(xmlname, imgname, targname, widinfo, proc_tp)


def main():
    parser = argparse.ArgumentParser("generate-scheme")
    parser.add_argument(
        "--config-dir",
        default="./saved_iam_data",
        help="file containing config CSVs",
    )
    parser.add_argument("--output-dir", default="./saved_iam_data", help="output dir")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)
    args = parser.parse_args()
    ####
    print(__file__, "with torch", torch.__version__)

    # create save directories
    setup_logging(args)
    torch.cuda.empty_cache()

    CTX.mldict.update(load_models(args))
    CTX.mldict["args"] = args
    with open("utils/char_placing.json", "r") as fp:
        CTX.mldict["cpj"] = json.load(fp)
    IAM_TempLoader.check_preload()

    ####
    pieces = ["clref", "qmreal", "qnreal", "qmfake", "qnfake", "qinterp"]
    for p in pieces:
        fname = os.path.join(args.config_dir, f"samp-{p}.csv")
        targdir = os.path.join(args.output_dir, p)
        os.makedirs(targdir, exist_ok=True)
        process_csv(fname, targdir)


if __name__ == "__main__":
    main()
