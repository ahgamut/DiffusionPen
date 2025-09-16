import numpy as np
import torch
from PIL import Image, ImageOps
import os
import glob
import json
import string
from torch.utils.data import Dataset

#
from utils.auxilary_functions import (
    image_resize_PIL,
    centered_PIL,
)
from utils.subprompt import Prompt, Word


def line_of_word(word):
    return word.id.split("-")[2]


def iam_resizefix(img_s):
    (img_width, img_height) = img_s.size
    img_s = img_s.resize((int(img_width * 64 / img_height), 64))
    (img_width, img_height) = img_s.size

    if img_width < 256:
        outImg = ImageOps.pad(
            img_s, size=(256, 64), color="white"
        )  # , centering=(0,0)) uncommment to pad right
        img_s = outImg

    else:
        # reduce image until width is smaller than 256
        while img_width > 256:
            img_s = image_resize_PIL(img_s, width=img_width - 20)
            (img_width, img_height) = img_s.size
        img_s = centered_PIL(img_s, (64, 256), border_value=255.0)

    return img_s


def get_word_data(word, img):
    orig = img.crop((word.x_start, word.y_start, word.x_end, word.y_end))
    rsz = iam_resizefix(orig)
    wraw = word.raw

    orig = np.array(orig, dtype=np.float32)
    rsz = np.array(rsz, dtype=np.float32)
    res = {"text": wraw, "image": rsz}
    # res["orig"] = orig
    return res


def read_iam_image(img_id):
    splits = img_id.split("-")
    p0 = splits[0]
    p1 = "-".join(splits[:2])
    path = os.path.join("./iam_data", "words", p0, p1, f"{img_id}.png")
    img = Image.open(path).convert("RGB")
    return img


def get_spacing_pairs(prompt, img):
    result = []
    for i in range(len(prompt.words) - 1):
        cur_word = prompt.words[i]
        next_word = prompt.words[i + 1]
        if line_of_word(cur_word) != line_of_word(next_word):
            continue

        try:
            cur_img = read_iam_image(cur_word.id)
            next_img = read_iam_image(next_word.id)
        except Exception as e:
            print("failed to read word image", e)

        # can add a check here for dupes
        cur_data = (cur_word.x_start, cur_word.y_start, cur_word.x_end, cur_word.y_end)
        next_data = (
            next_word.x_start,
            next_word.y_start,
            next_word.x_end,
            next_word.y_end,
        )

        diff_x = next_word.x_start - cur_word.x_end
        diff_y = next_word.y_start - cur_word.y_end
        coeffs = {
            "writer_id": prompt.writer_id,
            "cur_id": cur_word.id,
            "next_id": next_word.id,
            "cur_word": cur_word.raw,
            "next_word": next_word.raw,
            "cur_height": cur_word.height,
            "diff_x": diff_x,
            "diff_y": diff_y,
        }
        result.append(coeffs)
    return result


class IAMPlacerDataset(Dataset):
    STYLE_CLASSES = 339

    def __init__(
        self, basefolder="./iam_data", savefolder="./saved_iam_data", transforms=None
    ):
        self.basefolder = basefolder
        self.savefolder = savefolder
        self.transforms = transforms
        self.num_pairs = -1
        self.finalize()

    def __len__(self):
        return self.num_pairs

    def read_image(self, img_id):
        img = read_iam_image(img_id)
        return iam_resizefix(img)

    def __getitem__(self, index):
        wid = self.word_pairs["writer_id"][index]
        cur_word = self.word_pairs["cur_word"][index]
        next_word = self.word_pairs["next_word"][index]
        cur_id = self.word_pairs["cur_id"][index]
        next_id = self.word_pairs["next_id"][index]
        diff_x = self.word_pairs["diff_x"][index]
        diff_y = self.word_pairs["diff_y"][index]
        cur_height = self.word_pairs["cur_height"][index]

        x_cur = {"image": self.read_image(cur_id), "text": cur_word}
        x_next = {"image": self.read_image(next_id), "text": next_word}
        diff_tens = torch.tensor(
            [diff_x / cur_height, diff_y / cur_height],
            dtype=torch.float32,
            requires_grad=False,
        )

        if self.transforms is not None:
            x_cur["image"] = self.transforms(x_cur["image"])
            x_next["image"] = self.transforms(x_next["image"])
        return wid, x_cur, x_next, diff_tens

    def collate_fn(self, batch):
        wid, x_cur, x_next, diffs = zip(*batch)
        batch_wid = torch.stack(wid)
        #
        batch_cur = dict()
        batch_cur["text"] = [z["text"] for z in x_cur]
        batch_cur["image"] = torch.stack([z["image"] for z in x_cur])
        #
        batch_next = dict()
        batch_next["text"] = [z["text"] for z in x_next]
        batch_next["image"] = torch.stack([z["image"] for z in x_next])
        #
        batch_diffs = torch.stack(diffs)
        return batch_wid, batch_cur, batch_next, batch_diffs

    def finalize(self):
        save_file = os.path.join(self.savefolder, "placer_IAM.pt")
        if os.path.isfile(save_file):
            raw = torch.load(save_file, weights_only=False)  # unsafe, but just ndarrays
            print("loaded save file", save_file)
        else:
            raw = self.main_loader()
            torch.save(raw, save_file)

        self.word_pairs = raw
        self.num_pairs = len(self.word_pairs["cur_word"])
        print(f"dataset has {self.num_pairs} pairs")

    def main_loader(self):
        result = dict()
        xml_files = glob.glob(os.path.join(self.basefolder, "xml", "*.xml"))
        img_folder = os.path.join(self.basefolder, "forms")

        res_keys = [
            "cur_word",
            "cur_id",
            "next_word",
            "next_id",
            "writer_id",
            "diff_x",
            "diff_y",
            "cur_height",
        ]
        for k in res_keys:
            result[k] = []
        for fname in xml_files:
            print(len(result["cur_word"]))
            try:
                prompt = Prompt(fname)
                img = Image.open(os.path.join(img_folder, f"{prompt.id}.png"))
                img = img.convert("RGB")
                tmp = get_spacing_pairs(prompt, img)
                for x in tmp:
                    for k in res_keys:
                        result[k].append(x[k])
            except Exception as e:
                print(f"failed with {fname}", e)
        return result
