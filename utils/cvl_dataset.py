from os.path import isfile
from PIL import Image, ImageOps
from skimage import io as img_io
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision.utils import save_image
import cv2
import io
import json
import numpy as np
import os
import random
import string
import torch
import tqdm

#
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import (
    image_resize_PIL,
    centered_PIL,
    get_default_character_classes,
)

class CVLSDataset(Dataset):
    STYLE_CLASSES = 310

    def __init__(
        self,
        basefolder,
        subset,
        segmentation_level,
        fixed_size,
        tokenizer,
        text_encoder,
        feat_extractor,
        transforms,
        character_classes=None,
    ):
        super().__init__()
        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.stopwords = []
        self.stopwords_path = None
        if character_classes is None:
            self.character_classes = get_default_character_classes()
        else:
            self.character_classes = character_classes
        self.max_transcr_len = 0
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.feat_extractor = feat_extractor
        #
        self.setname = "CVL"
        self.trainset_file = "utils/splits_words/cvl_train_val.txt"
        self.valset_file = "utils/splits_words/cvl_val.txt"
        self.testset_file = "utils/splits_words/cvl_test.txt"
        self.windexmap_file = "utils/splits_words/writers_dict_cvl.json"
        self.finalize()

    def finalize(self):
        assert self.setname is not None
        if self.stopwords_path is not None:
            for line in open(self.stopwords_path):
                self.stopwords.append(line.strip().split(","))
            self.stopwords = self.stopwords[0]

        save_path = "./saved_iam_data"
        save_file = "{}/{}_{}_{}.pt".format(
            save_path, self.subset, self.segmentation_level, self.setname
        )
        if isfile(save_file):
            raw = torch.load(save_file, weights_only=False)  # safety issue
            print("loaded save file", save_file)
        else:
            raw = self.main_loader(self.subset, self.segmentation_level)
            torch.save(raw, save_file)

        self.img_paths = raw["paths"]
        self.data = raw["data"]
        self.wmap = raw["wmap"]
        self.windex_forward = json.load(open(self.windexmap_file))  # writer id to 0-309
        self.windex_backward = {
            v: k for k, v in self.windex_forward.items()
        }  # 0-309 to writer_id
        self.initial_writer_ids = [d[2] for d in self.data]
        self.writer_ids = list(self.wmap.keys())
        self.wclasses = len(self.writer_ids)
        print("Number of writers in", self.segmentation_level, ":", self.wclasses)

        # compute character classes given input transcriptions
        res = set(self.character_classes)
        res.add(" ")
        for _, transcr, _ in tqdm(self.data):
            res.update(list(transcr))
            self.max_transcr_len = max(self.max_transcr_len, len(transcr))
        res = sorted(list(res))
        print("Character classes: {} ({} different characters)".format(res, len(res)))
        print("Max transcription length: {}".format(self.max_transcr_len))
        self.character_classes = res
        self.max_transcr_len = self.max_transcr_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0].convert("RGB")
        transcr = self.data[index][1]
        wid = self.data[index][2]
        img_path = self.img_paths[index]

        # pick another sample that has the same writer id
        positive_samples = random.sample(self.wmap[wid], k=5)
        style_images = [x[0] for x in positive_samples]
        # ??
        cor_images = random.sample(self.wmap[wid], k=1)

        # transform
        img = self.transforms(img)
        cor_image = self.transforms(cor_images[0][0])
        s_imgs = torch.stack([self.transforms(x) for x in style_images])

        widi = self.windex_forward[wid]  # 0-309
        # why return image path?
        return img, transcr, widi, s_imgs, img_path, cor_im

    def collate_fn(self, batch):
        img, transcr, wid, s_imgs, img_path, cor_im = zip(*batch)
        # transcr_batch = torch.stack(transcr)
        # char_tokens_batch = torch.stack(char_tokens)
        img = torch.stack(img)
        transcr = torch.stack(transcr)
        s_imgs = torch.stack(s_imgs)
        cor_im = torch.stack(cor_im)
        return img, transcr, wid, s_imgs, img_path, cor_im

    @staticmethod
    def load_splits_text(fname):
        with open(fname, "r") as f:
            lines = f.readlines()
        result = []
        for l in lines:
            spl = l.split(",")
            rel_path = str(spl[0])
            writer_id = str(spl[1])
            transcr = str(",".join(spl[2:]))
            result.append((rel_path, transcr, writer_id))
        return result

    @staticmethod
    def fix_transcriptions(old_transcr):
        # transform iam transcriptions
        transcr = old_transcr.replace(" ", "")
        # "We 'll" -> "We'll"
        special_cases = ["s", "d", "ll", "m", "ve", "t", "re"]
        # lower-case
        for cc in special_cases:
            transcr = transcr.replace("|'" + cc, "'" + cc)
            transcr = transcr.replace("|'" + cc.upper(), "'" + cc.upper())
        transcr = transcr.replace("|", " ")
        return transcr

    def main_loader(self, subset, segmentation_level) -> list:
        if subset == "train":
            valid_set = CVLStyleDataset.load_splits_text(self.trainset_file)
        elif subset == "val":
            valid_set = CVLStyleDataset.load_splits_text(self.valset_file)
        elif subset == "test":
            valid_set = CVLStyleDataset.load_splits_text(self.testset_file)
        else:
            raise ValueError("can't pick subset")

        data = []
        paths = []
        wmap = dict()
        for i, (rel_path, transcr, writer_id) in enumerate(valid_set):
            img_path = os.path.join(self.basefolder, rel_path)
            img = Image.open(img_path).convert("RGB")  # .convert('L')
            if img.height < self.fixed_size[0] and img.width < self.fixed_size[1]:
                img = img
            else:
                img = image_resize_PIL(img, height=img.height // 2)

            transcr = CVLStyleDataset.fix_transcriptions(transcr)
            obj = (img, transcr, writer_id)
            if writer_id in wmap.keys():
                wmap[writer_id].append(i)
            else:
                wmap[writer_id] = [i]
            # data[i] = obj
            data.append(obj)
            paths.append(rel_path)

        raw = {"data": data, "paths": paths, "wmap": wmap}
        return raw
