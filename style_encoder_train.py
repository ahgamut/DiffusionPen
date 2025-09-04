import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import cross_entropy
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageOps
from os.path import isfile
from skimage import io
from torchvision.utils import save_image
from skimage.transform import resize
import os
import argparse
import torch.optim as optim
import timm
import cv2
import time
import json
import random

#
from utils.word_dataset import LineListIO
from utils.style_dataset import (
    WordStyleDataset,
    WLStyleDataset,
    IAMStyleDataset,
)
from utils.cvl_dataset import CVLStyleDataset
from utils.auxilary_functions import (
    affine_transformation,
    image_resize_PIL,
    centered_PIL,
)
from models import ImageEncoder, Mixed_Encoder, AvgMeter


# ================ Performance and Loss Function ========================
def performance(pred, label):
    # loss = nn.CrossEntropyLoss()
    # loss = loss(pred, label)
    loss = cross_entropy(pred, label)
    return loss


# ===================== Training ==========================================


def train_class_epoch(model, training_data, optimizer, args):
    """Epoch operation in training phase"""

    model.train()
    total_loss = 0
    n_corrects = 0
    total = 0
    pbar = training_data
    for i, data in enumerate(pbar):

        image = data[0].to(args.device)
        label = data[2].to(args.device)

        optimizer.zero_grad()

        output = model(image)

        loss = performance(output, label)
        _, preds = torch.max(output.data, 1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += label.size(0)
        n_corrects += (preds == label).sum().item()
        # pbar.set_postfix(Loss=loss.item())

    loss = total_loss / total
    accuracy = n_corrects / total

    return loss, accuracy


def eval_class_epoch(model, validation_data, args):
    """Epoch operation in evaluation phase"""

    model.eval()

    total_loss = 0
    total = 0
    n_corrects = 0
    prediction_list = []
    results = []
    with torch.no_grad():
        for i, data in enumerate(validation_data):

            image = data[0].to(args.device)
            image_paths = data[4]
            label = data[2].to(args.device)

            output = model(image)

            loss = performance(output, label)  # performance
            _, preds = torch.max(output.data, 1)

            total_loss += loss.item()
            n_corrects += (preds == label.data).sum().item()
            total += label.size(0)
            # prediction_list.append(preds)
            # write into a file the img_path and the prediction
            # with open('predictions.txt', 'a') as f:
            #     for i, p in enumerate(preds):
            #         f.write(f'{image_paths[i]},{p}\n')

    loss = total_loss / total
    accuracy = n_corrects / total

    return loss, accuracy


########################################################################
def train_epoch_triplet(train_loader, model, criterion, optimizer, device, args):
    model.train()
    running_loss = 0
    total = 0
    loss_meter = AvgMeter()
    pbar = train_loader
    for i, data in enumerate(pbar):

        img = data[0]
        wid = data[2]
        # print('wid', wid)
        positive = data[3]
        negative = data[4]

        anchor = img.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        # pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        # pbar.set_postfix(triplet_loss=loss_meter.avg)
        total += img.size(0)

    print("total", total)
    print("Training Loss: {:.4f}".format(running_loss / len(train_loader)))
    return running_loss / total  # np.mean(running_loss)/total


def val_epoch_triplet(val_loader, model, criterion, optimizer, device, args):
    running_loss = 0
    total = 0
    pbar = val_loader
    for i, data in enumerate(pbar):

        img = data[0]
        # transcr = data[1]
        wid = data[2]
        positive = data[3]
        negative = data[4]

        anchor = img.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)

        # running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        # pbar.set_postfix(triplet_loss=loss.item())
        total += wid.size(0)

    print("total", total)
    print("Validation Loss: {:.4f}".format(running_loss / len(val_loader)))
    return running_loss / total  # np.mean(running_loss)/total


############################ MIXED TRAINING ############################################
def train_epoch_mixed(
    train_loader,
    model,
    criterion_triplet,
    criterion_classification,
    optimizer,
    device,
    args,
):

    model.train()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    loss_meter_triplet = AvgMeter()
    loss_meter_class = AvgMeter()
    pbar = train_loader
    for i, data in enumerate(pbar):
        img = data[0]
        wid = data[2].to(device)
        positive = data[3].to(device)
        negative = data[4].to(device)

        anchor = img.to(device)
        # Get logits and features from the model
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)

        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()

        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(
            anchor_features, positive_features, negative_features
        )

        loss = classification_loss + triplet_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        # pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        loss_meter_triplet.update(triplet_loss.item(), count)
        loss_meter_class.update(classification_loss.item(), count)
        # pbar.set_postfix(
        #    mixed_loss=loss_meter.avg,
        #    classification_loss=loss_meter_class.avg,
        #    triplet_loss=loss_meter_triplet.avg,
        # )
        total += img.size(0)

    accuracy = n_corrects / total
    print("total", total)
    print("Training Loss: {:.4f}".format(running_loss / len(train_loader)))
    print("Training Accuracy: {:.4f}".format(accuracy * 100))
    return running_loss / total  # np.mean(running_loss)/total


def val_epoch_mixed(
    val_loader,
    model,
    criterion_triplet,
    criterion_classification,
    optimizer,
    device,
    args,
):

    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    pbar = val_loader
    for i, data in enumerate(pbar):

        img = data[0].to(device)
        wid = data[2].to(device)
        positive = data[3].to(device)
        negative = data[4].to(device)

        anchor = img
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)

        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()

        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(
            anchor_features, positive_features, negative_features
        )

        loss = classification_loss + triplet_loss

        # running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        # pbar.set_postfix(mixed_loss=loss_meter.avg)
        total += wid.size(0)

    print("total", total)
    accuracy = n_corrects / total
    print("Validation Loss: {:.4f}".format(running_loss / len(val_loader)))
    print("Validation Accuracy: {:.4f}".format(accuracy * 100))
    return running_loss / total  # np.mean(running_loss)/total


# TRAINING CALLS


def train_mixed(
    model,
    train_loader,
    val_loader,
    criterion_triplet,
    criterion_classification,
    optimizer,
    scheduler,
    device,
    args,
):
    best_loss = float("inf")
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_mixed(
            train_loader,
            model,
            criterion_triplet,
            criterion_classification,
            optimizer,
            device,
            args,
        )
        print("Epoch: {}/{}".format(epoch_i + 1, args.epochs))

        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_mixed(
                val_loader,
                model,
                criterion_triplet,
                criterion_classification,
                optimizer,
                device,
                args,
            )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                f"{args.save_path}/mixed_{args.dataset}_{args.model}.pth",
            )
            print("Saved Best Model!")

        scheduler.step(val_loss)


def train_classification(
    model, training_data, validation_data, optimizer, scheduler, device, args
):  # scheduler # after optimizer
    """Start training"""

    valid_accus = []
    num_of_no_improvement = 0
    best_acc = 0

    for epoch_i in range(args.epochs):
        print("[Epoch", epoch_i, "]")

        start = time.time()

        train_loss, train_acc = train_class_epoch(model, training_data, optimizer, args)
        print(
            "Training: {loss: 8.5f} , accuracy: {accu:3.3f} %, "
            "elapse: {elapse:3.3f} min".format(
                loss=train_loss, accu=100 * train_acc, elapse=(time.time() - start) / 60
            )
        )

        start = time.time()
        model_state_dict = model.state_dict()
        checkpoint = {"model": model_state_dict, "settings": args, "epoch": epoch_i}

        if validation_data is not None:
            val_loss, val_acc = eval_class_epoch(model, validation_data, args)
            print(
                "Validation: {loss: 8.5f} , accuracy: {accu:3.3f} %, "
                "elapse: {elapse:3.3f} min".format(
                    loss=val_loss, accu=100 * val_acc, elapse=(time.time() - start) / 60
                )
            )

            if val_acc > best_acc:

                print("- [Info] The checkpoint file has been updated.")
                best_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f"{args.save_path}/{args.dataset}_classification_{args.model}.pth",
                )
                num_of_no_improvement = 0
            else:
                num_of_no_improvement += 1

            if num_of_no_improvement >= 10:

                print("Early stopping criteria met, stopping...")
                break
        else:
            torch.save(
                model.state_dict(),
                f"{args.save_path}/{args.dataset}_classification_{args.model}.pth",
            )

        scheduler.step()


###


def train_triplet(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, args
):
    best_loss = float("inf")
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_triplet(
            train_loader, model, criterion, optimizer, device, args
        )
        print("Epoch: {}/{}".format(epoch_i + 1, args.epochs))

        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_triplet(
                val_loader, model, criterion, optimizer, device, args
            )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                f"{args.save_path}/triplet_{args.dataset}_{args.model}.pth",
            )
            print("Saved Best Model!")

        scheduler.step(val_loss)


### BUILDING DATASETS


def build_IAMDataset(args):
    dataset_folder = args.data_path
    aug_transforms = [lambda x: affine_transformation(x, s=0.1)]
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_data = IAMStyleDataset(
        dataset_folder,
        "train",
        "word",
        fixed_size=(1 * 64, 256),
        transforms=train_transform,
    )

    # split with torch.utils.data.Subset into train and val
    validation_size = int(0.2 * len(train_data))

    # Calculate the size of the training set
    train_size = len(train_data) - validation_size

    # Use random_split to split the dataset into train and validation sets
    train_data, val_data = random_split(
        train_data,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    print("len train data", len(train_data))
    print("len val data", len(val_data))

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    if val_loader is None:
        print("No validation data")

    style_classes = 339
    return train_data, val_data, train_loader, val_loader, style_classes


def build_CVLDataset(args):
    dataset_folder = args.data_path
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_data = CVLStyleDataset(
        basefolder=dataset_folder,
        subset="train",
        fixed_size=(1 * 64, 256),
        transforms=train_transform,
    )

    # split with torch.utils.data.Subset into train and val
    validation_size = int(0.2 * len(train_data))

    # Calculate the size of the training set
    train_size = len(train_data) - validation_size

    # Use random_split to split the dataset into train and validation sets
    train_data, val_data = random_split(
        train_data,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    print("len train data", len(train_data))
    print("len val data", len(val_data))

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    style_classes = CVLStyleDataset.STYLE_CLASSES
    return train_data, val_data, train_loader, val_loader, style_classes


###


def load_pretrained_weights(model, device, pretrained, style_path):
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )
    if pretrained == True:
        assert style_path != "", "need to provide style_path"
        state_dict = torch.load(style_path, map_location=device, weights_only=True)
        model_dict = model.state_dict()
        sub_dict = dict()
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                sub_dict[k] = v
            else:
                print("skipping pretrained weights for: ", k)
        model_dict.update(sub_dict)
        model.load_state_dict(model_dict)
        print("Pretrained model loaded")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Style Encoder")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv2_100",
        help="type of cnn to use (resnet, densenet, etc.)",
    )
    parser.add_argument("--dataset", type=str, default="iam", help="dataset name")
    parser.add_argument("--data-path", default="./iam_data", help="path to data")
    parser.add_argument(
        "--batch_size", type=int, default=320, help="input batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        required=False,
        help="number of training epochs",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use of feature extractor or not",
    )
    parser.add_argument("--style-path", default="./style_models", help="style path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to use for training / testing",
    )
    parser.add_argument(
        "--save-path", type=str, default="./style_models", help="path to save models"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        help="mixed for DiffusionPen, triplet for DiffusionPen-triplet, or classification for DiffusionPen-triplet",
    )
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ========= Data augmentation and normalization for training =====#
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)

    if args.dataset == "iam":
        train_data, val_data, train_loader, val_loader, style_classes = (
            build_IAMDataset(args)
        )
    elif args.dataset == "cvl":
        train_data, val_data, train_loader, val_loader, style_classes = (
            build_CVLDataset(args)
        )
    else:
        raise RuntimeError(
            "You need to add your own dataset and define the number of style classes!!!"
        )

    if args.model == "mobilenetv2_100":
        print("Using mobilenetv2_100")
        model = Mixed_Encoder(
            model_name="mobilenetv2_100",
            num_classes=style_classes,
            pretrained=True,
            trainable=True,
        )

    elif args.model == "resnet18":
        print("Using resnet18")
        model = Mixed_Encoder(
            model_name=args.model,
            num_classes=style_classes,
            pretrained=True,
            trainable=True,
        )

    else:
        raise RuntimeError("unknown model!")

    load_pretrained_weights(model, args.device, args.pretrained, args.style_path)
    model = model.to(device)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode="min", patience=3, factor=0.1
    )
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # print(model)
    # THIS IS THE CONDITION FOR DIFFUSIONPEN
    if args.mode == "mixed":
        criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        print("Using both classification and metric learning training")
        train_mixed(
            model,
            train_loader,
            val_loader,
            criterion_triplet,
            None,
            optimizer_ft,
            scheduler,
            device,
            args,
        )

    elif args.mode == "triplet":
        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer_ft,
            lr_scheduler,
            device,
            args,
        )

    elif args.mode == "classification":
        train_classification(
            model, train_loader, val_loader, optimizer_ft, scheduler, device, args
        )
    print("finished training")


if __name__ == "__main__":
    main()
