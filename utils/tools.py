import math
import os
import random
import tomllib
from typing import List, Tuple, Dict, Optional, Union, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

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


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


def config_update(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
    for k, v in (extra or {}).items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            config_update(base[k], v)
        else:
            base[k] = v


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG: Dict[str, Any] = load_config(os.path.join(_MODULE_DIR, "dataset_config.toml"))


def get_mean_std(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    dataset_name = dataset_name.upper()
    mean, std = CONFIG["MeanAndStd"][dataset_name]
    return mean, std


def get_normalizer(dataset_name: str) -> transforms.Normalize:
    mean, std = get_mean_std(dataset_name)
    normalizer = transforms.Normalize(mean=mean, std=std)
    return normalizer


def get_w_and_h(
    original_data: Optional[str], target_data: str
) -> Union[Tuple[int, int], Tuple[int, int, int, int]]:
    if original_data is None:
        w2, h2 = CONFIG["WidthAndHeight"][target_data.upper()]
        return w2, h2
    else:
        w1, h1 = CONFIG["WidthAndHeight"][original_data.upper()]
        w2, h2 = CONFIG["WidthAndHeight"][target_data.upper()]
        if w2 >= w1 and h2 >= h1:
            padding = int(w1 * 0.2)
            w2 = w1 - padding * 2
            h2 = h1 - padding * 2

        return w1, h1, w2, h2


class ImagenetteFilter(Dataset):
    """Wraps the Imagenette dataset (ImageFolder layout) with optional class filtering.

    Expected on-disk layout:
        {root}/imagenette2/train/{wnid}/...
        {root}/imagenette2/val/{wnid}/...

    The 10 WordNet IDs are sorted alphabetically by ImageFolder, giving stable
    class indices 0-9.  ``selected_classes`` refers to these indices.
    """

    SUBFOLDER = "imagenette2"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        selected_classes: Optional[List[int]] = None,
    ) -> None:
        self.selected_classes: List[int] = selected_classes or list(range(10))
        self.transform = transform
        split = "train" if train else "val"
        folder = os.path.join(root, self.SUBFOLDER, split)
        self.base_dataset = datasets.ImageFolder(root=folder, transform=None)
        self.indices = [
            i for i, (_, label) in enumerate(self.base_dataset.samples)
            if label in self.selected_classes
        ]
        self.label_map = {
            original: new for new, original in enumerate(self.selected_classes)
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, original_label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, self.label_map[original_label]


class CIFAR10Filter(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        selected_classes: Optional[List[int]] = None,
    ) -> None:
        self.selected_classes: List[int] = selected_classes or list(range(10))
        self.transform = transform
        self.base_dataset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=None
        )
        self.indices = [
            i for i, label in enumerate(self.base_dataset.targets)
            if label in self.selected_classes
        ]
        self.label_map = {
            original: new for new, original in enumerate(self.selected_classes)
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, original_label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, self.label_map[original_label]


class GTSRBFilter(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        selected_classes: Optional[List[int]] = None,
    ) -> None:
        self.selected_classes: List[int] = selected_classes or list(range(10))
        self.transform = transform
        self.base_dataset = datasets.GTSRB(
            root=root, split=split, download=True, transform=None
        )
        self._create_filtered_indices()
        self._create_label_mapping()

        if len(self.selected_classes) < 43:
            print(
                f"Filtered GTSRB ({split}): {len(self)} images from classes {self.selected_classes}"
            )

    def _create_filtered_indices(self) -> None:
        labels = [item[1] for item in self.base_dataset._samples]
        self.indices = [
            idx for idx, label in enumerate(labels) if label in self.selected_classes
        ]

    def _create_label_mapping(self) -> None:
        self.label_map = {
            original: new for new, original in enumerate(self.selected_classes)
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        img, original_label = self.base_dataset[original_idx]
        new_label = self.label_map[original_label]
        if self.transform:
            img = self.transform(img)
        return img, new_label


def get_dataset(
    dataset_name: str,
    normalize: bool = True,
    data_path: Optional[str] = None,
    transform: Optional[transforms.Compose] = None,
    train: bool = True,
    augmentation: bool = False,
    selected_classes: Optional[List[int]] = None,
) -> Dataset:
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
    if data_path is None:
        data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if dataset_name == "CIFAR10":
        if selected_classes is not None:
            return CIFAR10Filter(
                root=data_path, train=train, transform=transform,
                selected_classes=selected_classes,
            )
        return datasets.CIFAR10(
            root=data_path, train=train, download=True, transform=transform
        )

    elif dataset_name == "MNIST":
        dataset_class = datasets.MNIST
        return dataset_class(
            root=data_path, train=train, download=True, transform=transform
        )

    elif dataset_name == "CIFAR100":
        dataset_class = datasets.CIFAR100
        return dataset_class(
            root=data_path, train=train, download=True, transform=transform
        )

    elif dataset_name == "GTSRB":
        split = "train" if train else "test"
        return GTSRBFilter(
            root=data_path,
            split=split,
            transform=transform,
            selected_classes=selected_classes,
        )

    elif dataset_name == "SVHN":
        split = "train" if train else "test"
        return datasets.SVHN(
            root=data_path, split=split, download=True, transform=transform
        )

    elif dataset_name == "STL10":
        split = "train" if train else "test"
        return datasets.STL10(
            root=data_path, split=split, download=True, transform=transform
        )

    elif dataset_name == "TINYIMAGENET":
        folder_name = "train" if train else "val"
        return datasets.ImageFolder(
            root=os.path.join(data_path, f"tinyimagenet/{folder_name}"),
            transform=transform,
        )

    elif dataset_name == "IMAGENET100":
        folder_name = "train" if train else "val"
        return datasets.ImageFolder(
            root=os.path.join(data_path, f"imagenet100/{folder_name}"),
            transform=transform,
        )

    elif dataset_name == "LOCAL":
        folder_name = "train" if train else "test"
        return datasets.ImageFolder(
            root=os.path.join(data_path, f"LOCAL/{folder_name}"), transform=transform
        )

    elif dataset_name == "IMAGENETTE":
        return ImagenetteFilter(
            root=data_path,
            train=train,
            transform=transform,
            selected_classes=selected_classes,
        )

    else:
        raise ValueError("Unknown dataset")


def get_classnames(
    dataset_name: str, is_tuple: bool = False
) -> Union[List[str], Tuple[str, ...]]:
    dataset_name = dataset_name.upper()
    names = CONFIG["ClassNames"][dataset_name]
    if is_tuple:
        names = tuple(names)
    return names


def get_num_classes(dataset_name: str) -> int:
    dataset_name = dataset_name.upper()
    num_classes = CONFIG["NumClasses"][dataset_name]
    return num_classes


def load_model(
    num_classes: int,
    device: Union[str, torch.device],
    model_path: Optional[str] = None,
    model_arch: str = "RESNET18",
) -> nn.Module:
    model_arch = model_arch.upper()
    model: nn.Module

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
    elif model_arch in ["VGG16", "VGG16_BN", "VGG"]:
        model = archs.vgg.vgg16_bn(num_classes=num_classes)
    elif model_arch in ["VGG19", "VGG19_BN"]:
        model = archs.vgg.vgg19_bn(num_classes=num_classes)
    elif model_arch in ["VGG13", "VGG13_BN"]:
        model = archs.vgg.vgg13_bn(num_classes=num_classes)
    elif model_arch in ["VGG11", "VGG11_BN"]:
        model = archs.vgg.vgg11_bn(num_classes=num_classes)
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
    def __init__(
        self, dataset_name: str = "CIFAR10", device: Union[str, torch.device] = "cpu"
    ) -> None:
        self.dataset_name = dataset_name
        self.device = device
        self.mean, self.std = get_mean_std(dataset_name=self.dataset_name)

    def denormalize(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        return img * std + mean

    def visualize(self, data_path: str, num_samples: int = 10) -> None:
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
