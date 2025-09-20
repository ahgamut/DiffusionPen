import torch
from PIL import Image, ImageOps
import os
import glob
from torch.utils.data import Dataset
import struct
from dataclasses import dataclass

#
from utils.auxilary_functions import (
    image_resize_PIL,
    centered_PIL,
)
from utils.subprompt import Prompt, Word

#


@dataclass(frozen=True, slots=True)
class RelWordIndices:
    cur_index: int
    next_index: int

    @classmethod
    def from_bytes(cls, blob):
        return RelWordIndices(*struct.unpack("II", blob))

    def to_bytes(self):
        raw = struct.pack(
            "II",
            self.cur_index,
            self.next_index,
        )
        return raw


def line_of_word(word):
    return word.idd.split("-")[2]


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


def get_wimg_crop(word, img, resize=True, encode=True):
    orig = img.crop((word.x_start, word.y_start, word.x_end, word.y_end))
    if resize:
        rszd = iam_resizefix(orig)
    else:
        rszd = orig
    if encode:
        rszd = rszd.tobytes()
    return rszd


def read_iam_image(img_id):
    splits = img_id.split("-")
    p0 = splits[0]
    p1 = "-".join(splits[:2])
    path = os.path.join("./iam_data", "words", p0, p1, f"{img_id}.png")
    img = Image.open(path).convert("RGB")
    return img


def get_spacing_info(prompt, img, ind_start):
    pairs = []
    words = [w.to_bytes() for w in prompt.words]
    wimgs = [get_wimg_crop(w, img) for w in prompt.words]
    for i in range(len(prompt.words) - 1):
        cur_word = prompt.words[i]
        next_word = prompt.words[i + 1]
        assert cur_word.writer_id == next_word.writer_id
        if cur_word.parent_line != next_word.parent_line:
            continue

        cur_index = ind_start + i
        next_index = ind_start + i + 1
        rwi = RelWordIndices(cur_index, next_index)
        pairs.append(rwi.to_bytes())

    result = dict()
    result["words"] = words
    result["wimgs"] = wimgs
    result["pairs"] = pairs
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

    def read_image(self, index):
        raw = self.wimgs[index]
        img = Image.frombytes(mode="RGB", size=(256, 64), data=raw)
        return img

    def __getitem__(self, index):
        rwi = self.word_pairs[index]
        cur_word = self.words[rwi.cur_index]
        next_word = self.words[rwi.next_index]
        # finalize has checked that wids are same
        wid = cur_word.writer_id
        diff_x = next_word.x_start - cur_word.x_end
        diff_y = next_word.y_start - cur_word.y_end
        cur_height = cur_word.height

        x_cur = {"image": self.read_image(rwi.cur_index), "text": cur_word.raw}
        x_next = {"image": self.read_image(rwi.next_index), "text": next_word.raw}
        diff_tens = torch.tensor(
            [diff_y / cur_height],
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

        self.word_pairs = [RelWordIndices.from_bytes(x) for x in raw["pairs"]]
        self.wimgs = raw["wimgs"]
        self.words = [Word.from_bytes(x) for x in raw["words"]]
        self.num_pairs = len(self.word_pairs)
        self.validate_pairs()

    def validate_pairs(self):
        err_fmt = "wids? {} != {} (index={} := ({}, {})"
        for index, rwi in enumerate(self.word_pairs):
            cur_word = self.words[rwi.cur_index]
            next_word = self.words[rwi.next_index]
            err_string = err_fmt.format(
                cur_word.writer_id,
                next_word.writer_id,
                index,
                rwi.cur_index,
                rwi.next_index,
            )
            assert cur_word.writer_id == next_word.writer_id, err_string
        print(f"dataset has {self.num_pairs} pairs")

    def main_loader(self):
        result = dict()
        xml_files = glob.glob(os.path.join(self.basefolder, "xml", "*.xml"))
        img_folder = os.path.join(self.basefolder, "forms")

        res_keys = [
            "words",
            "wimgs",
            "pairs",
        ]
        for k in res_keys:
            result[k] = []
        for fname in xml_files:
            print(len(result["pairs"]), len(result["words"]))
            try:
                prompt = Prompt(fname)
                img = Image.open(os.path.join(img_folder, f"{prompt.idd}.png"))
                img = img.convert("RGB")
                tmp = get_spacing_info(prompt, img, len(result["words"]))
                for k in res_keys:
                    for x in tmp[k]:
                        result[k].append(x)
            except Exception as e:
                print(f"failed with {fname}", e)
        return result
