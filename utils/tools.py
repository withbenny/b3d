import math
import os
import random
import tomllib

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from . import archs


def get_unique_filename(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    new_path = f"{base}_{i}{ext}"
    while os.path.exists(new_path):
        i += 1
        new_path = f"{base}_{i}{ext}"
    return new_path


def load_config(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


def config_update(base: dict, extra: dict):
    for k, v in (extra or {}).items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            config_update(base[k], v)
        else:
            base[k] = v


CONFIG = load_config("./utils/dataset_config.toml")


def get_mean_std(dataset_name):
    dataset_name = dataset_name.upper()
    mean, std = CONFIG["MeanAndStd"][dataset_name]
    return mean, std


def get_normalizer(dataset_name):
    mean, std = get_mean_std(dataset_name)
    normalizer = transforms.Normalize(mean=mean, std=std)
    return normalizer


def get_w_and_h(original_data, target_data):
    if original_data is None:
        w2, h2 = CONFIG["WidthAndHeight"][target_data.upper()]
        return w2, h2
    else:
        w1, h1 = CONFIG["WidthAndHeight"][original_data.upper()]
        w2, h2 = CONFIG["WidthAndHeight"][target_data.upper()]
        if w2 >= w1 and h2 >= h1:
            padding = int(w1 * 0.125)
            w2 = w1 - padding * 2
            h2 = h1 - padding * 2

        return w1, h1, w2, h2


def get_dataset(
    dataset_name,
    normalize=True,
    data_path=None,
    transform=None,
    train=True,
    augmentation=False,
):
    dataset_name = dataset_name.upper()
    if transform is None:
        mean, std = get_mean_std(dataset_name)
        w, h = get_w_and_h(None, dataset_name)
        if augmentation:
            transforms_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=(h, w), padding=4),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize((h, w)),
                transforms.ToTensor(),
            ]
        if normalize:
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
        transform = transforms.Compose(transforms_list)
    if data_path == None:
        data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if dataset_name == "CIFAR10":
        train_set = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform
        )
    elif dataset_name == "MNIST":
        train_set = datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            root=data_path, train=False, download=True, transform=transform
        )
    elif dataset_name == "CIFAR100":
        train_set = datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform
        )
    elif dataset_name == "GTSRB":
        train_set = datasets.ImageFolder(
            root=os.path.join(data_path, "/GTSRB/train"), transform=transform
        )
        test_set = datasets.ImageFolder(
            root=os.path.join(data_path, "/GTSRB/test"), transform=transform
        )
    elif dataset_name == "STL10":
        train_set = datasets.STL10(
            root=data_path, split="train", download=True, transform=transform
        )
        test_set = datasets.STL10(
            root=data_path, split="test", download=True, transform=transform
        )
    elif dataset_name == "TINYIMAGENET":
        train_set = datasets.ImageFolder(
            root=os.path.join(data_path, "/tinyimagenet/train"), transform=transform
        )
        test_set = datasets.ImageFolder(
            root=os.path.join(data_path, "/tinyimagenet/val"), transform=transform
        )
    elif dataset_name == "IMAGENET100":
        train_set = datasets.ImageFolder(
            root=os.path.join(data_path, "/imagenet100/train"), transform=transform
        )
        test_set = datasets.ImageFolder(
            root=os.path.join(data_path, "/imagenet100/val"), transform=transform
        )
    elif dataset_name == "LOCAL":
        train_set = datasets.ImageFolder(
            root=os.path.join(data_path, "/LOCAL/train"), transform=transform
        )
        test_set = datasets.ImageFolder(
            root=os.path.join(data_path, "/LOCAL/test"), transform=transform
        )
    else:
        raise ValueError("Unknown dataset")

    if train:
        return train_set
    else:
        return test_set


def get_classnames(dataset_name, is_tuple=False):
    dataset_name = dataset_name.upper()
    names = CONFIG["ClassNames"][dataset_name]
    if is_tuple:
        names = tuple(names)
    return names


def get_num_classes(dataset_name):
    dataset_name = dataset_name.upper()
    num_classes = CONFIG["NumClasses"][dataset_name]
    return num_classes


def load_model(num_classes, device, model_path=None, model_arch="RESNET18"):
    model_arch = model_arch.upper()
    if model_arch == "RESNET18" or model_arch == "RESNET":
        model = archs.resnet.ResNet18(num_classes=num_classes)
    elif model_arch == "RESNET34":
        model = archs.resnet.ResNet34(num_classes=num_classes)
    elif model_arch == "RESNET50":
        model = archs.resnet.ResNet50(num_classes=num_classes)
    elif model_arch == "RESNET101":
        model = archs.resnet.ResNet101(num_classes=num_classes)
    elif model_arch == "RESNET152":
        model = archs.resnet.ResNet152(num_classes=num_classes)
    elif model_arch == "MOBILENETV2":
        model = archs.mobilenetv2.mobilenetv2(num_classes=num_classes)
    elif model_arch == "WRESNET":
        model = archs.wresnet.WideResNet(depth=28, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name")
    if model_path is None:
        return model
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        for param in model.parameters():
            param.requires_grad = False
        return model


class DataVisualizer:
    def __init__(self, dataset_name="CIFAR10", device="cpu"):
        self.dataset_name = dataset_name
        self.device = device
        self.mean, self.std = get_mean_std(dataset_name=self.dataset_name)

    def denormalize(self, img):
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        return img * std + mean

    def visualize(self, data_path, num_samples=10):
        data = torch.load(data_path, map_location=self.device)
        images = data["images"]
        labels = data["labels"]

        indices = random.sample(range(len(images)), num_samples)
        cols = min(num_samples, 5)
        rows = math.ceil(num_samples / cols)

        plt.figure(figsize=(cols * 4, rows * 4))
        for i, idx in enumerate(indices):
            img = images[idx]
            label = labels[idx]
            img = self.denormalize(img)
            np_img = img.permute(1, 2, 0).numpy()

            plt.subplot(rows, cols, i + 1)
            plt.imshow(np_img)
            plt.title(f"Label: {label}")
            plt.axis("off")
        plt.show()
