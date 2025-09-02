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

#
from utils.word_dataset import LineListIO
from utils.auxilary_functions import (
    affine_transformation,
    image_resize_PIL,
    centered_PIL,
)


class WordStyleDataset(Dataset):  # dummy
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


class WLStyleDataset(Dataset):
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
        # self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", )

    def __finalize__(self):
        """
        Will call code after descendant class has specified 'key' variables
        and ran dataset-specific code
        """
        assert self.setname is not None
        if self.stopwords_path is not None:
            for line in open(self.stopwords_path):
                self.stopwords.append(line.strip().split(","))
            self.stopwords = self.stopwords[0]

        save_path = "./IAM_dataset_PIL_style"
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        save_file = "{}/{}_{}_{}.pt".format(
            save_path, self.subset, self.segmentation_level, self.setname
        )  # dataset_path + '/' + set + '_' + level + '_IAM.pt'
        print("save_file", save_file)
        # if isfile(save_file) is False:
        #    data = self.main_loader(self.subset, self.segmentation_level)
        #    torch.save(data, save_file)   #Uncomment this in 'release' version
        # else:
        #    data = torch.load(save_file)

        data = self.main_loader(self.subset, self.segmentation_level)
        self.data = data
        # print('data', self.data)
        self.initial_writer_ids = [d[2] for d in data]
        writer_ids, _ = np.unique([d[2] for d in data], return_inverse=True)
        self.writer_ids = writer_ids
        self.wclasses = len(writer_ids)
        print("Number of writers", self.wclasses)
        if self.character_classes is None:
            res = set()
            # compute character classes given input transcriptions
            for _, transcr, _, _ in tqdm(data):
                # print('legth transcr = ', len(transcr))
                res.update(list(transcr))
                self.max_transcr_len = max(self.max_transcr_len, len(transcr))
                # print('self.max_transcr_len', self.max_transcr_len)

            res = sorted(list(res))
            res.append(" ")
            print(
                "Character classes: {} ({} different characters)".format(res, len(res))
            )
            print("Max transcription length: {}".format(self.max_transcr_len))
            self.character_classes = res
            self.max_transcr_len = self.max_transcr_len
        # END FINALIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        transcr = self.data[index][1]
        wid = self.data[index][2]
        img_path = self.data[index][3]
        # pick another sample that has the same self.data[2] or same writer id
        positive_samples = [p for p in self.data if p[2] == wid and len(p[1]) > 3]
        negative_samples = [n for n in self.data if n[2] != wid and len(n[1]) > 3]
        positive = random.choice(positive_samples)[0]

        # Make sure you have at least 5 matching images
        if len(positive_samples) >= 5:
            # Randomly select 5 indices from the matching_indices
            random_samples = random.sample(positive_samples, k=5)
            # Retrieve the corresponding images
            style_images = [i[0] for i in random_samples]
        else:
            # Handle the case where there are fewer than 5 matching images (if needed)
            # print("Not enough matching images with writer ID", wid)
            positive_samples_ = [p for p in self.data if p[2] == wid]
            # print('len positive samples', len(positive_samples_), 'wid', wid)
            random_samples_ = random.sample(positive_samples_, k=5)
            # Retrieve the corresponding images
            style_images = [i[0] for i in random_samples_]

        # pick another image from a different writer
        negative = random.choice(negative_samples)[0]

        img_pos = positive  # image_resize_PIL(positive, height=positive.height // 2)
        img_neg = negative  # image_resize_PIL(negative, height=negative.height // 2)

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

        # img = image_resize_PIL(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
        # img = centered_PIL(img, (fheight, fwidth), border_value=None).convert('L')
        # image = image.resize((256, 64), Image.ANTIALIAS)
        if img.width < 256:
            img = ImageOps.pad(
                img, size=(256, 64), color="white"
            )  # , centering=(0,0)) uncommment to pad right
        # print('img', img.mode, img.size)

        pixel_values_img = img  # self.processor(img, return_tensors="pt").pixel_values
        pixel_values_img = pixel_values_img  # .squeeze(0)

        img_pos = image_resize_PIL(
            img_pos, height=int(1.0 * nheight_pos), width=int(1.0 * nwidth_pos)
        )
        img_pos = centered_PIL(img_pos, (fheight, fwidth), border_value=255.0)

        img_neg = image_resize_PIL(
            img_neg, height=int(1.0 * nheight_neg), width=int(1.0 * nwidth_neg)
        )
        img_neg = centered_PIL(img_neg, (fheight, fwidth), border_value=255.0)

        pixel_values_pos = (
            img_pos  # self.processor(img_pos, return_tensors="pt").pixel_values
        )
        pixel_values_neg = (
            img_neg  # self.processor(img_neg, return_tensors="pt").pixel_values
        )
        pixel_values_pos = pixel_values_pos  # .squeeze(0)
        pixel_values_neg = pixel_values_neg  # .squeeze(0)

        st_imgs = []
        for s_im in style_images:
            # s_im = image_resize_PIL(s_im, height=s_im.height // 2)
            if self.subset == "train":
                nwidth = int(np.random.uniform(0.75, 1.25) * s_im.width)
                nheight = int(
                    (np.random.uniform(0.9, 1.1) * s_im.height / s_im.width) * nwidth
                )

            else:
                nheight, nwidth = s_im.height, s_im.width

            nheight, nwidth = max(4, min(fheight - 16, nheight)), max(
                8, min(fwidth - 32, nwidth)
            )
            # Load the image and transform it
            s_img = image_resize_PIL(
                s_im, height=int(1.0 * nheight), width=int(1.0 * nwidth)
            )
            s_img = centered_PIL(s_img, (fheight, fwidth), border_value=255.0)
            if self.transforms is not None:
                s_img_tensor = self.transforms(img)

            st_imgs += [s_img_tensor]

        s_imgs = torch.stack(st_imgs)

        if self.transforms is not None:
            img = self.transforms(img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)

        char_tokens = [self.character_classes.index(c) for c in transcr]
        # print('char_tokens before', char_tokens)
        pad_token = 79

        # padding_length = self.max_transcr_len - len(char_tokens)
        padding_length = 95 - len(char_tokens)
        char_tokens.extend([pad_token] * padding_length)

        # char_tokens += [pad_token] * (self.max_transcr_len - len(char_tokens))
        char_tokens = torch.tensor(char_tokens, dtype=torch.long)

        cla = self.character_classes
        # print('character classes', cla)
        # wid = self.wr_dict[index]
        # print('wid after', index, wid)
        # print('pixel_values_pos', pixel_values_pos.shape)
        # img = outImg
        # save_image(img, 'check_augm.png')
        return (
            img,
            transcr,
            char_tokens,
            wid,
            img_pos,
            img_neg,
            cla,
            s_imgs,
            img_path,
            img,
            img_pos,
            img_neg,
        )  # pixel_values_img, pixel_values_pos, pixel_values_neg

    def collate_fn(self, batch):
        # Separate image tensors and caption tensors
        (
            img,
            transcr,
            char_tokens,
            wid,
            positive,
            negative,
            cla,
            s_imgs,
            img_path,
            pixel_values_img,
            pixel_values_pos,
            pixel_values_neg,
        ) = zip(*batch)

        # Stack image tensors and caption tensors into batches
        images_batch = torch.stack(img)
        # transcr_batch = torch.stack(transcr)
        char_tokens_batch = torch.stack(char_tokens)

        images_pos = torch.stack(positive)
        images_neg = torch.stack(negative)

        s_imgs = torch.stack(s_imgs)

        pixel_values_img = torch.stack(pixel_values_img)
        pixel_values_pos = torch.stack(pixel_values_pos)
        pixel_values_neg = torch.stack(pixel_values_neg)

        return (
            img,
            transcr,
            char_tokens_batch,
            wid,
            images_pos,
            images_neg,
            cla,
            s_imgs,
            img_path,
            pixel_values_img,
            pixel_values_pos,
            pixel_values_neg,
        )

    def main_loader(self, subset, segmentation_level) -> list:
        # This function should be implemented by an inheriting class.
        raise NotImplementedError

    def check_size(self, img, min_image_width_height, fixed_image_size=None):
        """
        checks if the image accords to the minimum and maximum size requirements
        or fixed image size and resizes if not

        :param img: the image to be checked
        :param min_image_width_height: the minimum image size
        :param fixed_image_size:
        """
        if fixed_image_size is not None:
            if len(fixed_image_size) != 2:
                raise ValueError("The requested fixed image size is invalid!")
            new_img = resize(
                image=img, output_shape=fixed_image_size[::-1], mode="constant"
            )
            new_img = new_img.astype(np.float32)
            return new_img
        elif np.amin(img.shape[:2]) < min_image_width_height:
            if np.amin(img.shape[:2]) == 0:
                print("OUCH")
                return None
            scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
            new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
            new_img = resize(image=img, output_shape=new_shape, mode="constant")
            new_img = new_img.astype(np.float32)
            return new_img
        else:
            return img


class IAMStyleDataset(WLStyleDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = "IAM"
        self.trainset_file = "{}/set_split/trainset.txt".format(self.basefolder)
        self.valset_file = "{}/set_split/validationset1.txt".format(self.basefolder)
        self.testset_file = "{}/set_split/testset.txt".format(self.basefolder)
        self.line_file = "{}/ascii/lines.txt".format(self.basefolder)
        self.word_file = "{}/ascii/words.txt".format(self.basefolder)
        self.word_path = "{}/words".format(self.basefolder)
        self.line_path = "{}/lines".format(self.basefolder)
        self.forms = "{}/ascii/forms.txt".format(self.basefolder)
        # self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        def gather_iam_info(self, set="train", level="word"):
            if subset == "train":
                # valid_set = np.loadtxt(self.trainset_file, dtype=str)
                valid_set = np.loadtxt(
                    "./utils/aachen_iam_split/train_val.uttlist", dtype=str
                )
                # print(valid_set)
            elif subset == "val":
                # valid_set = np.loadtxt(self.valset_file, dtype=str)
                valid_set = np.loadtxt(
                    "./utils/aachen_iam_split/validation.uttlist", dtype=str
                )
            elif subset == "test":
                # valid_set = np.loadtxt(self.testset_file, dtype=str)
                valid_set = np.loadtxt(
                    "./utils/aachen_iam_split/test.uttlist", dtype=str
                )
            else:
                raise ValueError
            if level == "word":
                gtfile = self.word_file
                root_path = self.word_path
                print("root_path", root_path)
                forms = self.forms
            elif level == "line":
                gtfile = self.line_file
                root_path = self.line_path
            else:
                raise ValueError
            gt = []
            form_writer_dict = {}

            dict_path = f"utils/writers_dict_{subset}.json"
            # open dict file
            with open(dict_path, "r") as f:
                wr_dict = json.load(f)
            for l in open(forms):
                if not l.startswith("#"):
                    info = l.strip().split()
                    # print('info', info)
                    form_name = info[0]
                    writer_name = info[1]
                    form_writer_dict[form_name] = writer_name
                    # print('form_writer_dict', form_writer_dict)
                    # print('form_name', form_name)
                    # print('writer', writer_name)

            for line in open(gtfile):
                if not line.startswith("#"):
                    info = line.strip().split()
                    name = info[0]
                    name_parts = name.split("-")
                    pathlist = [root_path] + [
                        "-".join(name_parts[: i + 1]) for i in range(len(name_parts))
                    ]
                    # print('name', name)
                    # form =
                    # writer_name = name_parts[1]
                    # print('writer_name', writer_name)

                    if level == "word":
                        line_name = pathlist[-2]
                        del pathlist[-2]

                        if info[1] != "ok":
                            continue

                    elif level == "line":
                        line_name = pathlist[-1]
                    form_name = "-".join(line_name.split("-")[:-1])
                    # print('form_name', form_name)
                    # if (info[1] != 'ok') or (form_name not in valid_set):
                    if form_name not in valid_set:
                        # print(line_name)
                        continue
                    img_path = "/".join(pathlist)

                    transcr = " ".join(info[8:])
                    writer_name = form_writer_dict[form_name]
                    # print('writer_name', writer_name)
                    writer_name = wr_dict[writer_name]

                    gt.append((img_path, transcr, writer_name))
            return gt

        info = gather_iam_info(self, subset, segmentation_level)
        data = []
        for i, (img_path, transcr, writer_name) in enumerate(info):
            if i % 1000 == 0:
                print(
                    "imgs: [{}/{} ({:.0f}%)]".format(
                        i, len(info), 100.0 * i / len(info)
                    )
                )
            #

            try:
                img = Image.open(img_path + ".png").convert("RGB")  # .convert('L')
                if img.height < 64 and img.width < 256:
                    img = img
                else:
                    img = image_resize_PIL(img, height=img.height // 2)

            except:
                print("Could not add image file {}.png".format(img_path))
                continue

            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case
            for cc in special_cases:
                transcr = transcr.replace("|'" + cc, "'" + cc)
                transcr = transcr.replace("|'" + cc.upper(), "'" + cc.upper())

            transcr = transcr.replace("|", " ")
            data.append((img, transcr, writer_name, img_path))

        return data
