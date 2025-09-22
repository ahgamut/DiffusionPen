from os.path import isfile
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import json
import numpy as np
import os
import random
import string
import torch
from tqdm import tqdm

#
from utils.auxilary_functions import (
    image_resize_PIL,
    centered_PIL,
    get_default_character_classes,
)


class CVLBaseDataset(Dataset):
    STYLE_CLASSES = 310

    def __init__(
        self,
        basefolder,
        subset,
        segmentation_level,
        fixed_size,
        transforms,
        character_classes=None,
        args=None,
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
        self.args = args
        #
        self.setname = "CVL"
        self.trainset_file = "utils/splits_words/cvl_train_val.txt"
        self.valset_file = "utils/splits_words/cvl_val.txt"
        self.testset_file = "utils/splits_words/cvl_test.txt"
        self.fullset_file = "utils/splits_words/cvl_full.txt"
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
            raw = torch.load(save_file, weights_only=False)  # unsafe, but just ndarrays
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
        self.character_classes = res
        self.max_transcr_len = self.max_transcr_len

    def samples_from_index(self, ind, num_samples, is_widi=False, strict=True):
        if is_widi:
            # widi is in 0-309
            wid = self.windex_backward[ind]
        else:
            wid = ind

        positives = self.wmap[wid]
        imgs = []
        paths = []
        while len(paths) < num_samples:
            mas = random.sample(positives, num_samples)
            for ma_ind in mas:
                ma = self.data[ma_ind]
                if len(ma[2]) > 3 or not strict:
                    imgs.append(ma[0])
                    paths.append(self.img_paths[ma_ind])

        result = {"imgs": imgs[:num_samples], "paths": paths[:num_samples]}
        return result

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

    def main_loader(self, subset, segmentation_level):
        if subset == "train":
            valid_set = CVLBaseDataset.load_splits_text(self.trainset_file)
        elif subset == "val":
            valid_set = CVLBaseDataset.load_splits_text(self.valset_file)
        elif subset == "test":
            valid_set = CVLBaseDataset.load_splits_text(self.testset_file)
        elif subset == "full":
            valid_set = CVLBaseDataset.load_splits_text(self.fullset_file)
        else:
            raise ValueError("can't pick subset")

        data = []
        paths = []
        wmap = dict()
        for i, (rel_path, transcr, writer_id) in enumerate(valid_set):
            print(i)
            img_path = os.path.join(self.basefolder, rel_path)
            img = Image.open(img_path).convert("RGB")  # .convert('L')
            transcr = CVLBaseDataset.fix_transcriptions(transcr)
            if transcr in string.punctuation:
                img = centered_PIL(img, (64, 256), border_value=255.0)
            else:
                (img_width, img_height) = img.size
                # resize image to height 64 keeping aspect ratio
                img = img.resize((int(img_width * 64 / img_height), 64))
                (img_width, img_height) = img.size

                while img_width > 256:
                    img = image_resize_PIL(img, width=img_width - 20)
                    (img_width, img_height) = img.size

                if img_width < 256:
                    outImg = ImageOps.pad(
                        img,
                        size=(256, 64),
                        centering=(0.5, 0.5),
                        color=255,
                    )
                    img = outImg

                img = centered_PIL(img, (64, 256), border_value=255.0)

            # convert to ndarray before storing
            img = np.array(img)

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


class CVLDataset(CVLBaseDataset):

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
        args=None,
    ):
        super().__init__(
            basefolder=basefolder,
            subset=subset,
            segmentation_level=segmentation_level,
            fixed_size=fixed_size,
            character_classes=character_classes,
            transforms=transforms,
            args=args,
        )
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.feat_extractor = feat_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        transcr = self.data[index][1]
        wid = self.data[index][2]
        img_path = self.img_paths[index]

        # pick another sample that has the same writer id
        style_samp = self.samples_from_index(wid, 5, is_widi=False, strict=True)
        style_images = style_samp["imgs"]
        # ??
        cor_image = self.samples_from_index(wid, 1, is_widi=False, strict=False)["imgs"]

        # transform
        img = self.transforms(img)
        cor_image = self.transforms(cor_image[0])
        s_imgs = torch.stack([self.transforms(x) for x in style_images])

        widi = self.windex_forward[wid]  # 0-309
        # why return image path?
        return img, transcr, widi, s_imgs, img_path, cor_image

    def collate_fn(self, batch):
        img, transcr, wid, s_imgs, img_path, cor_im = zip(*batch)
        # transcr_batch = torch.stack(transcr)
        # char_tokens_batch = torch.stack(char_tokens)
        img = torch.stack(img)
        transcr = torch.stack(transcr)
        s_imgs = torch.stack(s_imgs)
        cor_im = torch.stack(cor_im)
        return img, transcr, wid, s_imgs, img_path, cor_im


class CVLStyleDataset(CVLBaseDataset):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        transcr = self.data[index][1]
        wid = self.data[index][2]

        # pick another sample that has the same writer id
        positive_wid = wid
        samp_pos = self.samples_from_index(positive_wid, 1, is_widi=False, strict=True)
        img_pos = samp_pos["imgs"][0]

        # pick another image from a different writer
        negative_wid = random.choice(self.writer_ids)
        while negative_wid == wid:
            negative_wid = random.choice(self.writer_ids)
        samp_neg = self.samples_from_index(negative_wid, 1, is_widi=False, strict=True)
        img_neg = samp_neg["imgs"][0]

        if self.transforms is not None:
            img = self.transforms(img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)

        widi = self.windex_forward[wid]  # 0-309
        # why return image path?
        return img, transcr, widi, img_pos, img_neg, ""

    def collate_fn(self, batch):
        img, transcr, wid, positive, negative, img_path = zip(*batch)
        # transcr_batch = torch.stack(transcr)
        # char_tokens_batch = torch.stack(char_tokens)
        images_batch = torch.stack(img)
        images_pos = torch.stack(positive)
        images_neg = torch.stack(negative)
        return images_batch, transcr, wid, images_pos, images_neg, img_path
