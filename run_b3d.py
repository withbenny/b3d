"""
run_b3d.py — Batch B3D backdoor detection on CIFAR-10 ResNet-18 models.

Walks through every .pt model file under a root directory, runs B3D
detection on each one, and writes per-model JSON results plus a CSV/JSON
summary into a single `b3d_results/` folder.

Usage
-----
  python run_b3d.py [options]

  # minimal — use defaults (./models, ./data are symlinked to hard-label-detection)
  python run_b3d.py

  # with CUDA and custom iterations
  python run_b3d.py --model_dir ... --device cuda --iters 1000 --k 50

  # resume an interrupted run (skips models that already have a result file)
  python run_b3d.py --model_dir ... --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import datasets, transforms

# Make sure b3d.py, detect.py, and utils/ are importable from the same directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from detect import detect_backdoor, print_report
from utils.tools import load_model as _utils_load_model, get_normalizer, get_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLEAN_FOLDER_PREFIXES = {"clean"}

DATASET_CONFIGS = {
    "cifar10": {
        "num_classes":     10,
        "image_shape":     (3, 32, 32),
        "normalizer_name": "CIFAR10",
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
    },
    "gtsrb": {
        "num_classes":     43,
        "image_shape":     (3, 32, 32),
        "normalizer_name": "GTSRB",
        "class_names":     [f"class_{i}" for i in range(43)],
    },
    "cifar100": {
        "num_classes":     100,
        "image_shape":     (3, 32, 32),
        "normalizer_name": "CIFAR100",
        "class_names":     [f"class_{i}" for i in range(100)],
    },
    "tinyimagenet": {
        "num_classes":     200,
        "image_shape":     (3, 64, 64),
        "normalizer_name": "TINYIMAGENET",
        "class_names":     [f"class_{i}" for i in range(200)],
    },
}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model(model_path: Path, device: str, num_classes: int,
               model_arch: str = "RESNET18") -> nn.Module:
    """Load a model via utils.tools.load_model (uses the authoritative arch definitions)."""
    model = _utils_load_model(
        num_classes=num_classes,
        device=device,
        model_path=str(model_path),
        model_arch=model_arch,
    )
    model.to(device).eval()
    return model


def make_model_fn(model: nn.Module, device: str, normalizer_name: str) -> Callable:
    """
    Return a model_fn that moves images to `device`, applies dataset
    normalisation, and returns logits. B3D operates in raw [0,1] pixel
    space; normalisation lives here so the trigger optimisation never
    touches the standardised domain.
    """
    _normalize = get_normalizer(normalizer_name)

    @torch.no_grad()
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return model(_normalize(x.to(device)))

    return _fn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clean_images(
    dataset_name: str,
    data_root: str,
    n_images: int = 10000,
    seed: int = 0,
) -> torch.Tensor:
    """
    Return `n_images` randomly sampled test images in [0,1].
    Normalisation is NOT applied here — it lives inside model_fn.
    """
    # get_dataset handles Resize, ToTensor, and dataset-specific paths.
    # normalize=False because normalisation lives inside model_fn.
    ds = get_dataset(dataset_name, normalize=False, data_path=data_root, train=False)

    n_images = min(n_images, len(ds))
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=rng)[:n_images]
    images = torch.stack([ds[i.item()][0] for i in idx])
    logger.info("Loaded %d %s test images from %s", len(images), dataset_name.upper(), data_root)
    return images


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_models(model_dir: Path) -> List[Tuple[str, Path]]:
    """
    Walk `model_dir` and return (attack_type, model_path) pairs for every
    .pt file found in immediate subdirectories.

    Directory layout expected:
        model_dir/
          {attack_type}/
            full_base_aug_seed=XXXXXXXX.pt
            ...
    """
    results: List[Tuple[str, Path]] = []
    for attack_dir in sorted(model_dir.iterdir()):
        if not attack_dir.is_dir():
            continue
        # Skip result folders and other non-attack directories
        if attack_dir.name in ("b3d_results", "xray_results"):
            continue
        for pt_file in sorted(attack_dir.glob("*.pt")):
            results.append((attack_dir.name, pt_file))
    return results


def is_clean_model(attack_type: str) -> bool:
    """Return True if the folder name indicates a non-backdoored model."""
    name_lower = attack_type.lower()
    return any(name_lower.startswith(prefix) for prefix in CLEAN_FOLDER_PREFIXES)


# ---------------------------------------------------------------------------
# Per-model detection
# ---------------------------------------------------------------------------

def run_one_model(
    attack_type: str,
    model_path: Path,
    clean_images: torch.Tensor,
    dataset_cfg: Dict,
    detect_classes: int,
    model_arch: str,
    k: int,
    sigma: float,
    lr: float,
    lam: float,
    num_iterations: int,
    batch_size: int,
    device: str,
    resume: bool,
) -> Optional[Dict]:
    """
    Run B3D detection on a single model.

    Results are written to <attack_dir>/b3d_results/<model_stem>.json.
    Returns the serialisable record dict, or None if skipped (resume mode).
    """
    result_dir = model_path.parent / "b3d_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    stem = model_path.stem
    json_path = result_dir / f"{stem}.json"
    trigger_path = result_dir / f"{stem}_triggers.pt"

    if resume and json_path.exists():
        logger.info("SKIP (already done): %s", stem)
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    logger.info("=" * 60)
    logger.info("Model     : %s", stem)
    logger.info("Attack    : %s", attack_type)
    logger.info("=" * 60)

    t0 = time.time()
    try:
        model    = load_model(model_path, device, num_classes=dataset_cfg["num_classes"], model_arch=model_arch)
        model_fn = make_model_fn(model, device, normalizer_name=dataset_cfg["normalizer_name"])

        results = detect_backdoor(
            model_fn=model_fn,
            num_classes=detect_classes,
            image_shape=dataset_cfg["image_shape"],
            X=clean_images.clone(),
            use_synthetic=False,
            k=k,
            sigma=sigma,
            lr=lr,
            lam_init=lam,
            num_iterations=num_iterations,
            batch_size=batch_size,
            device=device,
            save_path=None,  # we save triggers below
        )

        elapsed = time.time() - t0

        # Save triggers separately (tensors are not JSON-serialisable)
        torch.save(
            {
                "masks":    [m.cpu() for m in results["masks"]],
                "patterns": [p.cpu() for p in results["patterns"]],
            },
            str(trigger_path),
        )

        # Serialisable summary (no tensors)
        record = {
            "model_path":       str(model_path),
            "attack_type":      attack_type,
            "model_stem":       model_path.stem,
            "ground_truth_backdoored": not is_clean_model(attack_type),
            "is_backdoored":    results["is_backdoored"],
            "suspected_classes": results["suspected_classes"],
            "suspected_class_names": [dataset_cfg["class_names"][c] for c in results["suspected_classes"]],
            "detect_classes":   detect_classes,
            "l1_norms":         results["l1_norms"],
            "anomaly_scores":   results["anomaly_scores"],
            "flags":            results["flags"],
            "elapsed_sec":      round(elapsed, 1),
            "trigger_path":     str(trigger_path),
            # hyper-params for reproducibility
            "hyperparams": {
                "k": k, "sigma": sigma, "lr": lr,
                "lam_init": lam, "num_iterations": num_iterations,
                "batch_size": batch_size,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        logger.info(
            "DONE  %s/%s  →  %s  (%.1f s)",
            attack_type, stem,
            "BACKDOORED" if results["is_backdoored"] else "CLEAN",
            elapsed,
        )
        print_report(results, dataset_cfg["class_names"])

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        return record

    except Exception as exc:
        logger.error("FAILED %s: %s", stem, exc, exc_info=True)
        error_record = {
            "model_path":  str(model_path),
            "attack_type": attack_type,
            "model_stem":  model_path.stem,
            "error":       str(exc),
        }
        with open(json_path.with_suffix(".error.json"), "w", encoding="utf-8") as f:
            json.dump(error_record, f, indent=2)
        return None


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def build_summary(records: List[Dict], model_dir: Path) -> None:
    """
    Write summaries at two levels:
      - Per-attack-type: <attack_dir>/b3d_results/summary.json
      - Overall:         <model_dir>/b3d_summary.csv  +  b3d_summary.json
    """
    if not records:
        logger.warning("No records to summarise.")
        return

    # ---- Group by attack type -----------------------------------------------
    by_attack: Dict[str, Dict] = {}
    for r in records:
        at = r["attack_type"]
        if at not in by_attack:
            by_attack[at] = {
                "total": 0,
                "ground_truth_backdoored": not is_clean_model(at),
                "detected_as_backdoored": 0,
                "correct": 0,
                "models": [],
            }
        if "error" in r:
            continue
        s    = by_attack[at]
        gt   = r["ground_truth_backdoored"]
        pred = r["is_backdoored"]
        l1   = r["l1_norms"]
        s["total"] += 1
        s["detected_as_backdoored"] += int(pred)
        s["correct"] += int(gt == pred)
        s["models"].append({
            "stem":              r["model_stem"],
            "is_backdoored":     pred,
            "suspected_classes": r["suspected_classes"],
            "min_l1":            min(l1),
            "median_l1":         float(np.median(l1)),
        })

    for s in by_attack.values():
        n = s["total"]
        s["accuracy"]       = round(s["correct"] / n, 4) if n else 0.0
        s["detection_rate"] = round(s["detected_as_backdoored"] / n, 4) if n else 0.0

    # ---- Per-attack-type summary.json ---------------------------------------
    for at, s in by_attack.items():
        at_result_dir = model_dir / at / "b3d_results"
        at_result_dir.mkdir(parents=True, exist_ok=True)
        path = at_result_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        logger.info("Per-attack summary → %s", path)

    # ---- Overall CSV --------------------------------------------------------
    valid = [r for r in records if "error" not in r]
    csv_lines = [
        "attack_type,model_stem,ground_truth_backdoored,is_backdoored,"
        "correct,suspected_classes,min_l1,median_l1,elapsed_sec\n"
    ]
    for r in valid:
        gt = r["ground_truth_backdoored"]
        l1 = r["l1_norms"]
        csv_lines.append(
            f"{r['attack_type']},"
            f"{r['model_stem']},"
            f"{gt},"
            f"{r['is_backdoored']},"
            f"{gt == r['is_backdoored']},"
            f"\"{r['suspected_classes']}\","
            f"{min(l1):.4f},"
            f"{float(np.median(l1)):.4f},"
            f"{r.get('elapsed_sec', '')}\n"
        )
    csv_path = model_dir / "b3d_summary.csv"
    csv_path.write_text("".join(csv_lines), encoding="utf-8")
    logger.info("Overall CSV → %s", csv_path)

    # ---- Overall JSON -------------------------------------------------------
    total   = len(valid)
    correct = sum(1 for r in valid if r["ground_truth_backdoored"] == r["is_backdoored"])
    overall = {
        "total_models":  total,
        "correct":       correct,
        "accuracy":      round(correct / total, 4) if total else 0.0,
        "failed_models": len(records) - total,
    }
    json_path = model_dir / "b3d_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "by_attack_type": by_attack}, f, indent=2)
    logger.info("Overall JSON → %s", json_path)

    # ---- Console ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  B3D BATCH DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total models : {total}  |  Correct : {correct}  |  Accuracy : {overall['accuracy']:.2%}")
    print("-" * 70)
    print(f"  {'Attack Type':<30} {'Total':>6} {'GT-BD':>6} {'Detected':>9} {'Correct':>8} {'Acc':>6}")
    print("-" * 70)
    for at, s in sorted(by_attack.items()):
        gt_str = "Yes" if s["ground_truth_backdoored"] else "No"
        print(
            f"  {at:<30} {s['total']:>6} {gt_str:>6} "
            f"{s['detected_as_backdoored']:>9} {s['correct']:>8} {s['accuracy']:>6.2%}"
        )
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch B3D backdoor detection on CIFAR-10 ResNet-18 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_dir",
        default=str(SCRIPT_DIR / "models" / "cifar10" / "target"),
        help="Root directory containing attack-type sub-folders with .pt model files.",
    )
    p.add_argument(
        "--data_root",
        default=str(SCRIPT_DIR / "data"),
        help="Directory containing the dataset folder.",
    )
    p.add_argument(
        "--dataset", default="cifar10", choices=list(DATASET_CONFIGS.keys()),
        help="Dataset name (cifar10 / gtsrb).",
    )
    p.add_argument(
        "--detect_classes", type=int, default=None,
        help="Only reverse-engineer the first N classes (default: all classes in dataset).",
    )
    p.add_argument(
        "--model_arch", default="RESNET18",
        help="Model architecture (RESNET18 / MOBILENETV2 / WRESNET / ...).",
    )
    p.add_argument(
        "--n_clean", type=int, default=10000,
        help="Number of test images used as clean data for B3D.",
    )
    # B3D hyperparameters (paper defaults)
    p.add_argument("--k",          type=int,   default=50,    help="NES samples k.")
    p.add_argument("--sigma",      type=float, default=0.1,   help="NES Gaussian std σ.")
    p.add_argument("--lr",         type=float, default=0.05,  help="Adam learning rate.")
    p.add_argument("--lam",        type=float, default=0.01,  help="Initial λ.")
    p.add_argument("--iters",      type=int,   default=1000,  help="Optimisation iterations T.")
    p.add_argument("--batch_size", type=int,   default=32,    help="Minibatch size.")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device.")
    p.add_argument(
        "--attack_types", nargs="*", default=None,
        help="Only process these attack-type folders (e.g. badnet_1.0_0.1 clean_1.0).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip models whose result .json file already exists.",
    )
    p.add_argument(
        "--data_seed", type=int, default=42,
        help="Random seed for sampling clean images.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    model_dir = Path(args.model_dir)

    dataset_cfg = DATASET_CONFIGS[args.dataset]

    detect_classes = args.detect_classes or dataset_cfg["num_classes"]

    logger.info("Model directory : %s", model_dir)
    logger.info("Device          : %s", args.device)
    logger.info("Dataset         : %s (%d classes, detecting %d)",
                args.dataset.upper(), dataset_cfg["num_classes"], detect_classes)

    # Load clean images once (reused for all models)
    clean_images = load_clean_images(args.dataset, args.data_root, args.n_clean, seed=args.data_seed)
    clean_images = clean_images.to(args.device)

    # Discover all models
    all_models = discover_models(model_dir)
    if args.attack_types:
        all_models = [(at, p) for at, p in all_models if at in args.attack_types]

    logger.info("Found %d model file(s) across %d attack type(s).",
                len(all_models), len({at for at, _ in all_models}))

    # Run detection on every model
    records: List[Dict] = []
    for i, (attack_type, model_path) in enumerate(all_models, 1):
        logger.info("[%d / %d]", i, len(all_models))
        rec = run_one_model(
            attack_type=attack_type,
            model_path=model_path,
            clean_images=clean_images,
            dataset_cfg=dataset_cfg,
            detect_classes=detect_classes,
            model_arch=args.model_arch,
            k=args.k,
            sigma=args.sigma,
            lr=args.lr,
            lam=args.lam,
            num_iterations=args.iters,
            batch_size=args.batch_size,
            device=args.device,
            resume=args.resume,
        )
        if rec is not None:
            records.append(rec)

    # Load any pre-existing JSON records (for --resume runs where some were skipped)
    if args.resume:
        for at_dir in sorted(model_dir.iterdir()):
            if not at_dir.is_dir() or at_dir.name == "b3d_results":
                continue
            for json_file in sorted((at_dir / "b3d_results").glob("*.json")):
                if json_file.name == "summary.json":
                    continue
                already = any(
                    r.get("model_stem") == json_file.stem and r.get("attack_type") == at_dir.name
                    for r in records
                )
                if not already:
                    with open(json_file, encoding="utf-8") as f:
                        try:
                            records.append(json.load(f))
                        except json.JSONDecodeError:
                            pass

    build_summary(records, model_dir)


if __name__ == "__main__":
    main()
