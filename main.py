import torch
import numpy as np
import os
import sys

from utils import tools
from b3d import B3D


def detector(config_path: str = "CONFIG.toml"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG = tools.load_config(config_path)

    dataset_name = CONFIG["B3D"]["dataset_name"]
    num_classes = tools.get_num_classes(dataset_name)
    model = tools.load_model(
        num_classes=num_classes,
        device=device,
        model_path=CONFIG["B3D"]["model_path"],
        model_arch=CONFIG["B3D"]["model_arch"],
    )
    model.to(device)
    model.eval()

    test_set = tools.get_dataset(
        dataset_name=dataset_name,
        data_path="./data",
        train=False,
        normalize=False,
        augmentation=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=CONFIG["B3D"]["Hyperparameters"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    normalizer = tools.get_normalizer(dataset_name)
    input_shape = test_set[0][0].shape

    detector = B3D(
        model=model,
        normalizer=normalizer,
        input_shape=input_shape,
        num_classes=num_classes,
        k=CONFIG["B3D"]["Hyperparameters"]["k"],
        lr=CONFIG["B3D"]["Hyperparameters"]["lr"],
        sigma=CONFIG["B3D"]["Hyperparameters"]["sigma"],
        device=device,
    )

    l1_norms = []
    for c in range(num_classes):
        print(f"Running B3D for target class {c}...")
        l1_norm = detector.run_detection(
            dataloader=test_loader,
            target_class=c,
            num_iterations=CONFIG["B3D"]["Hyperparameters"]["num_iterations"],
        )
        l1_norms.append(l1_norm)
        print(f"Class {c}: L1 norm of mask = {l1_norm:.4f}")

    l1_norms_np = np.array(l1_norms)
    median_norm = np.median(l1_norms_np)
    min_norm = l1_norms_np.min()
    min_class = l1_norms_np.argmin()
    threshold = median_norm / 4.0

    print("\n Detection Summary:")
    print(f" - Minimum L1 norm: {min_norm:.4f} (Class {min_class})")
    print(f" - Threshold: {threshold:.4f}")
    if min_norm <= threshold:
        print(f" --> Potential backdoor detected in class {min_class}!")
    else:
        print(" --> No backdoors detected.")


if __name__ == "__main__":
    detector()
