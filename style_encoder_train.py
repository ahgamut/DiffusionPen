import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
import argparse
import torch.optim as optim

#
from utils.word_dataset import MergedWordDataset
from models import Mixed_Encoder, AvgMeter


# ================ Performance and Loss Function ========================
def performance(pred, label):
    loss = cross_entropy(pred, label)
    return loss


############################ MIXED TRAINING ############################################
def train_epoch_mixed(train_loader, model, criterion_triplet, optimizer, device, args):

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

        running_loss += loss.item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        loss_meter_triplet.update(triplet_loss.item(), count)
        loss_meter_class.update(classification_loss.item(), count)
        total += img.size(0)

    accuracy = n_corrects / total
    print("total", total)
    print("Training Loss: {:.4f}".format(running_loss / len(train_loader)))
    print("Training Accuracy: {:.4f}".format(accuracy * 100))
    return running_loss / total  # np.mean(running_loss)/total


def val_epoch_mixed(val_loader, model, criterion_triplet, optimizer, device, args):

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

        running_loss += loss.item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
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

        scheduler.step()


### BUILDING DATASETS


def build_style_dataset(args):
    """Style-encoder pretraining data over the merged memmap split
    (``MergedWordDataset`` in style_mode -> anchor/positive/negative triplets).
    ``style_classes`` is the merged writer count W."""
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full = MergedWordDataset(
        args.data_dir,
        transforms=train_transform,
        args=args,
        style_mode=True,
    )
    style_classes = full.wclasses

    validation_size = int(0.2 * len(full))
    train_size = len(full) - validation_size
    train_data, val_data = random_split(
        full,
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
    return train_data, val_data, train_loader, val_loader, style_classes


###


def load_pretrained_weights(model, device, pretrained, style_path):
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )
    if pretrained:
        assert style_path != "", "need to provide style_path"
        style_state_dict = torch.load(style_path, map_location=device, weights_only=True)
        model_dict = model.state_dict()
        sub_dict = dict()
        for k, v in style_state_dict.items():
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
    parser.add_argument(
        "--data-dir", type=str,
        default="./saved_iam_data/combined_word_train",
        help="path to the merged dataset split directory built by "
        "utils/build_multidataset.py",
    )
    parser.add_argument(
        "--dataset", type=str, default="combined",
        help="tag used only in saved checkpoint filenames",
    )
    parser.add_argument(
        "--batch-size", type=int, default=320, help="input batch size for training"
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
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ========= Data augmentation and normalization for training =====#
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_data, val_data, train_loader, val_loader, style_classes = (
        build_style_dataset(args)
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

    # DiffusionPen style encoder: joint classification + triplet metric learning.
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    print("Using both classification and metric learning training")
    train_mixed(
        model,
        train_loader,
        val_loader,
        criterion_triplet,
        optimizer_ft,
        scheduler,
        device,
        args,
    )
    print("finished training")


if __name__ == "__main__":
    main()
