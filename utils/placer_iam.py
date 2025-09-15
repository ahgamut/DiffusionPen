import numpy as np
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


def get_line_of_word(word):
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
    rsz = iam_resizefix(wimg)
    wraw = word.raw

    orig = np.array(orig, dtype=np.float32)
    rsz = np.array(rsz, dtype=np.float32)
    return {"text": wraw, "orig": orig, "image": rsz}


def get_spacing_pairs(prompt, img):
    result = []
    for i in range(len(prompt.words) - 1):
        cur_word = prompt.words[i]
        next_word = prompt.words[i + 1]
        if line_of_word(cur_word) != line_of_word(next_word):
            continue

        # can add a check here for dupes
        cur_data = get_word_data(cur_word, img)
        next_data = get_word_data(next_word, img)

        diff_x = next_word.x_start - cur_word.x_end
        diff_y = next_word.y_start - cur_word.y_end
        diffs = {"cur_height": cur_word.height, "x": diff_x, "y": diff_y}
        result.append((prompt.writer_id, cur_data, next_data, diffs))
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

    def __getitem__(self, index):
        wid, x_cur, x_next, diffs = self.word_pairs[index]
        if self.transforms is not None:
            x_cur["image"] = self.transforms(x_cur["image"])
            x_next["image"] = self.transforms(x_next["image"])
        diff_tens = torch.tensor(
            [diffs["x"] / diffs["height"], diffs["y"] / diffs["height"]],
            dtype=torch.float32,
            requires_grad=False,
        )
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
            raw = self.main_loader(self.subset, self.segmentation_level)
            torch.save(raw, save_file)

        self.num_pairs = len(raw)
        self.word_pairs = raw
        print(f"dataset has {self.num_pairs} pairs")

    def main_loader(self):
        result = []
        xml_files = glob.glob(os.path.join(self.basefolder, "xml", "*.xml"))
        img_folder = os.path.join(self.basefolder, "forms")
        for fname in xml_files:
            try:
                prompt = Prompt(fname)
                img = Image.open(os.path.join(img_folder, f"{prompt.id}.png"))
                img = img.convert("RGB")
                result.append(get_spacing_pairs(prompt, img))
            except Exception as e:
                print(f"failed with {fname}", e)
        return result
