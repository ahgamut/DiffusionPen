import os
import json
import random
from PIL import Image, ImageOps
from torchvision import transforms

#
from utils.auxiliary_functions import (
    image_resize_PIL,
    centered_PIL,
)


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


class IAM_TempLoader:
    """Reads raw IAM word crops (``./iam_data/words``) for the generation
    scripts and the style-bank builder. This is a data-loading concern; it is
    intentionally kept out of the model definitions (``models/diffpen2.py``),
    whose inference path relies solely on the precomputed style bank."""

    wr_dict = None
    reverse_wr_dict = None
    train_data = None
    root_path = "./iam_data/words"
    wmap = None
    tform = None

    @classmethod
    def check_preload(cls):
        if cls.wr_dict is None:
            with open("utils/writers_dict_train_iam.json", "r") as f:
                cls.wr_dict = json.load(f)
                cls.reverse_wr_dict = {v: k for k, v in cls.wr_dict.items()}

        if cls.train_data is None:
            with open("./utils/splits_words/iam_train_val.txt", "r") as f:
                # with open('./utils/splits_words/iam_test.txt', 'r') as f:
                train_data = f.readlines()
                cls.train_data = [i.strip().split(",") for i in train_data]

        if cls.wmap is None:
            wmap = dict()
            for obj in cls.train_data:
                img_path = obj[0]
                wid = obj[1]
                transcr = ",".join(obj[2:])
                if wid in wmap.keys():
                    wmap[wid].append((img_path, wid, transcr))
                else:
                    wmap[wid] = [(img_path, wid, transcr)]
            cls.wmap = wmap

        if cls.tform is None:
            cls.tform = transforms.ToTensor()

    @classmethod
    def map_index_to_wid(cls, label_index):
        return cls.reverse_wr_dict[label_index]

    @classmethod
    def map_wid_to_index(cls, wid):
        return cls.wr_dict[wid]

    @classmethod
    def get_refs(cls, label_index, n_samples):
        wid = cls.map_index_to_wid(label_index)
        matching_lines = cls.wmap[wid]

        paths = []
        imgs = []
        while len(imgs) < 5:
            mas = random.sample(matching_lines, n_samples)
            for ma in mas:
                ma_path = None
                ma_img = None
                if len(ma[2]) > 3:
                    ma_path = os.path.join(cls.root_path, ma[0])
                if ma_path is not None:
                    try:
                        ma_img = Image.open(ma_path).convert("RGB")
                    except Exception:
                        # Handle the exception (e.g., print an error message)
                        print(f"Error loading image from {ma_path}")
                if ma_img is not None:
                    imgs.append(ma_img)
                    paths.append(ma[0])

        result = {"paths": paths[:5], "imgs": imgs[:5]}
        return result
