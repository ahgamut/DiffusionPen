from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageOps
from os.path import isfile
from skimage import io
from torchvision.utils import save_image
from skimage.transform import resize
import os
import json
import random


class WordStyleDataset(Dataset):
    #
    # TODO list:
    #
    #   Create method that will print data statistics (min/max pixel value, num of channels, etc.)
    """
    This class is a generic Dataset class meant to be used for word- and line- image datasets.
    It should not be used directly, but inherited by a dataset-specific class.
    """

    def __init__(
        self,
        basefolder: str = "datasets/",  # Root folder
        subset: str = "all",  # Name of dataset subset to be loaded. (e.g. 'all', 'train', 'test', 'fold1', etc.)
        segmentation_level: str = "line",  # Type of data to load ('line' or 'word')
        fixed_size: tuple = (128, None),  # Resize inputs to this size
        transforms: list = None,  # List of augmentation transform functions to be applied on each input
        character_classes: list = None,  # If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
    ):

        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.setname = None  # E.g. 'IAM'. This should coincide with the folder name
        self.stopwords = []
        self.stopwords_path = None
        self.character_classes = character_classes
        self.max_transcr_len = 0
        self.data_file = "./iam_data/iam_train_val_fixed.txt"

        with open(self.data_file, "r") as f:
            lines = f.readlines()

        self.data_info = [line.strip().split(",") for line in lines]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):

        img = self.data_info[index][0]
        img = Image.open(img).convert("RGB")
        transcr = self.data_info[index][2]

        wid = self.data_info[index][1]

        img_path = self.data_info[index][0]
        # pick another sample that has the same self.data[2] or same writer id
        positive_samples = [p for p in self.data_info if p[1] == wid and len(p[2]) > 3]
        negative_samples = [n for n in self.data_info if n[1] != wid and len(n[2]) > 3]

        # print('wid', wid)
        positive = random.choice(positive_samples)[0]

        # print('positive', positive)
        # pick another image from a different writer
        negative = random.choice(negative_samples)[0]
        # print('negative', negative)
        img_pos = Image.open(positive).convert(
            "RGB"
        )  # image_resize_PIL(positive, height=positive.height // 2)
        img_neg = Image.open(negative).convert(
            "RGB"
        )  # image_resize_PIL(negative, height=negative.height // 2)

        if img.height < 64 and img.width < 256:
            img = img
        else:
            img = image_resize_PIL(img, height=img.height // 2)

        if img_pos.height < 64 and img_pos.width < 256:
            img_pos = img_pos
        else:
            img_pos = image_resize_PIL(img_pos, height=img_pos.height // 2)

        if img_neg.height < 64 and img_neg.width < 256:
            img_neg = img_neg
        else:
            img_neg = image_resize_PIL(img_neg, height=img_neg.height // 2)

        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        # print('fheight', fheight, 'fwidth', fwidth)
        if self.subset == "train":
            nwidth = int(np.random.uniform(0.75, 1.25) * img.width)
            nheight = int(
                (np.random.uniform(0.9, 1.1) * img.height / img.width) * nwidth
            )

            nwidth_pos = int(np.random.uniform(0.75, 1.25) * img_pos.width)
            nheight_pos = int(
                (np.random.uniform(0.9, 1.1) * img_pos.height / img_pos.width)
                * nwidth_pos
            )

            nwidth_neg = int(np.random.uniform(0.75, 1.25) * img_neg.width)
            nheight_neg = int(
                (np.random.uniform(0.9, 1.1) * img_neg.height / img_neg.width)
                * nwidth_neg
            )

        else:
            nheight, nwidth = img.height, img.width
            nheight_pos, nwidth_pos = img_pos.height, img_pos.width
            nheight_neg, nwidth_neg = img_neg.height, img_neg.width

        nheight, nwidth = max(4, min(fheight - 16, nheight)), max(
            8, min(fwidth - 32, nwidth)
        )
        nheight_pos, nwidth_pos = max(4, min(fheight - 16, nheight_pos)), max(
            8, min(fwidth - 32, nwidth_pos)
        )
        nheight_neg, nwidth_neg = max(4, min(fheight - 16, nheight_neg)), max(
            8, min(fwidth - 32, nwidth_neg)
        )

        img = image_resize_PIL(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
        img = centered_PIL(img, (fheight, fwidth), border_value=255.0)

        img_pos = image_resize_PIL(
            img_pos, height=int(1.0 * nheight_pos), width=int(1.0 * nwidth_pos)
        )
        img_pos = centered_PIL(img_pos, (fheight, fwidth), border_value=255.0)

        img_neg = image_resize_PIL(
            img_neg, height=int(1.0 * nheight_neg), width=int(1.0 * nwidth_neg)
        )
        img_neg = centered_PIL(img_neg, (fheight, fwidth), border_value=255.0)

        if self.transforms is not None:

            img = self.transforms(img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)

        return img, transcr, wid, img_pos, img_neg, img_path

    def collate_fn(self, batch):
        # Separate image tensors and caption tensors
        img, transcr, wid, positive, negative, img_path = zip(*batch)

        # Stack image tensors and caption tensors into batches
        images_batch = torch.stack(img)
        # transcr_batch = torch.stack(transcr)
        # char_tokens_batch = torch.stack(char_tokens)

        images_pos = torch.stack(positive)
        images_neg = torch.stack(negative)

        return images_batch, transcr, wid, images_pos, images_neg, img_path
