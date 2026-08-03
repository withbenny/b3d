"""
detect.py — Backdoor detection and mitigation using reversed B3D triggers.

Implements:
  * Outlier detection via MAD + heuristic  (Section 3.3, Section 4)
  * Full detection pipeline  (B3D and B3D-SS variants)
  * Mitigation score  S(x) = D_KL(f(x) || f(A(x,m,p)))  (Section 5, Eq. 8)
  * CLI entry-point

Reference:
  Dong et al. (2021) "Black-box Detection of Backdoor Attacks with
  Limited Information and Data", ICCV 2021.
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from b3d import B3DOptimizer, apply_trigger, generate_synthetic_samples

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(l1_norms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify backdoored classes by spotting unusually small L1 norms.

    Two complementary criteria (paper Section 4, Outlier detection):

    1. MAD z-score: flag class c if |L1_c - median| / (1.4826 * MAD) > 2.
    2. Heuristic: flag class c if L1_c < median / 4.

    A class is suspected when either criterion fires.

    NOTE: L1 norms should be computed on the SOFT mask g(theta_m),
    not the binarised mask (see B3DOptimizer.optimize_class).

    Args:
        l1_norms: Soft-mask L1 norms per class, shape [C].

    Returns:
        flags:  Boolean array [C], True = suspected backdoor target class.
        scores: MAD anomaly z-score [C] (higher = more anomalous).
    """
    median = np.median(l1_norms)
    mad = np.median(np.abs(l1_norms - median))
    scores = np.abs(l1_norms - median) / (1.4826 * mad + 1e-10)

    heuristic = l1_norms < (median / 4.0)
    flags = (scores > 2.0) | heuristic
    return flags, scores


# ---------------------------------------------------------------------------
# Mitigation score  S(x) = D_KL(f(x) || f(A(x,m,p)))  Eq. (8)
# ---------------------------------------------------------------------------

def mitigation_score(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    masks: List[torch.Tensor],
    patterns: List[torch.Tensor],
    suspected_classes: List[int],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the mitigation score S(x) per input image (Eq. 8).

    S(x) = D_KL( f(x) || f(A(x, m, p)) )

    Large S -> predictions change a lot after stamping -> x is likely clean.
    Small S -> x may already carry the trigger.

    Returns scores [N]; higher score = more likely clean.
    """
    x = x.to(device)
    with torch.no_grad():
        p_x = torch.softmax(model_fn(x), dim=1)

    scores = torch.zeros(x.shape[0], device=device)
    for c in suspected_classes:
        m   = masks[c].to(device)
        pat = patterns[c].to(device)
        x_a = apply_trigger(x, m, pat)
        with torch.no_grad():
            p_xa = torch.softmax(model_fn(x_a), dim=1)
        kl = (p_x * (torch.log(p_x + 1e-10) - torch.log(p_xa + 1e-10))).sum(dim=1)
        scores = scores + kl

    if suspected_classes:
        scores = scores / len(suspected_classes)
    return scores


# ---------------------------------------------------------------------------
# Full detection pipeline
# ---------------------------------------------------------------------------

def detect_backdoor(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    num_classes: int,
    image_shape: Tuple[int, ...],
    X: Optional[torch.Tensor] = None,
    use_synthetic: bool = False,
    n_synthetic_per_class: int = 100,
    # --- B3D hyperparameters (paper defaults) ---
    k: int = 50,
    sigma: float = 0.1,
    lr: float = 0.05,
    lam_init: float = 0.01,
    lam_up: float = 1.2,
    lam_down: float = 0.8,
    lam_max: float = 1e2,
    lam_min: float = 1e-5,
    asr_window: int = 10,
    asr_threshold: float = 0.99,
    num_iterations: int = 1000,
    batch_size: int = 128,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> Dict:
    """
    End-to-end B3D / B3D-SS backdoor detection.

    Steps
    -----
    1. Optionally generate synthetic images (B3D-SS, Section 3.4).
    2. Run Algorithm 1 for every class c to reverse-engineer trigger (m_c, p_c).
    3. Detect outliers via MAD + heuristic on soft-mask L1 norms.

    Args:
        model_fn:              Black-box model — input float [N, *image_shape]
                               in [0,1], output logits [N, num_classes].
        num_classes:           C.
        image_shape:           Shape of one image, e.g. (3, 32, 32).
        X:                     Clean validation images [N, *image_shape] in [0,1].
                               Required when use_synthetic=False.
        use_synthetic:         Run B3D-SS (synthesise images, no real data needed).
        n_synthetic_per_class: Images per class for B3D-SS (paper: 100).
        k:                     NES samples (paper: 50).
        sigma:                 NES Gaussian std (paper: 0.1).
        lr:                    Adam learning rate (paper: 0.05).
        lam_init:              Initial L1 regularisation weight (paper: 0.01).
        lam_up:                Lambda multiplier when rolling-mean ASR >= threshold.
        lam_down:              Lambda multiplier when rolling-mean ASR < threshold.
        lam_max:               Hard upper bound on lambda.
        lam_min:               Hard lower bound on lambda.
        asr_window:            Rolling window size for ASR smoothing.
        asr_threshold:         Target ASR for adaptive lambda (paper: 0.99).
        num_iterations:        Optimisation steps T (paper: until convergence).
        batch_size:            Minibatch size (paper: 128).
        device:                Torch device string.
        save_path:             If provided, save triggers + results to this .pt file.

    Returns dict with:
        is_backdoored      bool
        suspected_classes  list[int]
        l1_norms           list[float]   -- soft-mask L1 norm per class
        anomaly_scores     list[float]   -- MAD z-score per class
        flags              list[bool]    -- outlier flag per class
        masks              list[Tensor]  -- soft masks g(theta_m) per class
        patterns           list[Tensor]  -- trigger patterns g(theta_p) per class
    """
    # ---- B3D-SS: synthesise dataset if no clean images are available ------
    if use_synthetic or X is None:
        logger.info("B3D-SS mode -- generating %d synthetic images per class ...",
                    n_synthetic_per_class)
        X = generate_synthetic_samples(
            model_fn=model_fn,
            num_classes=num_classes,
            image_shape=image_shape,
            n_per_class=n_synthetic_per_class,
            k=k, sigma=sigma, lr=lr,
            num_iterations=500,
            device=device,
        )
        logger.info("Synthetic dataset: %d images total.", X.shape[0])

    X = X.to(device)

    # ---- Algorithm 1: reverse-engineer triggers for all classes ------------
    opt = B3DOptimizer(
        model_fn=model_fn,
        num_classes=num_classes,
        image_shape=image_shape,
        k=k, sigma=sigma, lr=lr,
        lam_init=lam_init,
        lam_up=lam_up,
        lam_down=lam_down,
        lam_max=lam_max,
        lam_min=lam_min,
        asr_window=asr_window,
        asr_threshold=asr_threshold,
        num_iterations=num_iterations,
        batch_size=batch_size,
        device=device,
    )
    # run() returns (masks, patterns, l1_norms) where l1_norms are soft-mask L1
    masks, patterns, l1_list = opt.run(X)

    # ---- Outlier detection ------------------------------------------------
    l1 = np.array(l1_list, dtype=np.float64)
    flags, scores = detect_outliers(l1)
    suspected = [c for c in range(num_classes) if flags[c]]

    results: Dict = {
        "is_backdoored":    bool(flags.any()),
        "suspected_classes": suspected,
        "l1_norms":         l1.tolist(),
        "anomaly_scores":   scores.tolist(),
        "flags":            flags.tolist(),
        "masks":            masks,
        "patterns":         patterns,
    }

    if save_path:
        torch.save(
            {
                "masks":            [m.cpu() for m in masks],
                "patterns":         [p.cpu() for p in patterns],
                "l1_norms":         l1,
                "anomaly_scores":   scores,
                "suspected_classes": suspected,
                "is_backdoored":    results["is_backdoored"],
            },
            save_path,
        )
        logger.info("Triggers and results saved to %s", save_path)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: Dict, class_names: Optional[List[str]] = None) -> None:
    """Print a formatted per-class detection report to stdout."""
    C = len(results["l1_norms"])
    names = class_names or [str(c) for c in range(C)]
    median_l1 = float(np.median(results["l1_norms"]))

    verdict = "BACKDOORED" if results["is_backdoored"] else "CLEAN"
    print("\n" + "=" * 68)
    print("  B3D BACKDOOR DETECTION REPORT")
    print("=" * 68)
    print(f"  Verdict            : {verdict}")
    if results["suspected_classes"]:
        sus = [names[c] for c in results["suspected_classes"]]
        print(f"  Suspected class(es): {sus}")
    print(f"  Median soft-L1     : {median_l1:.3f}")
    print("-" * 68)
    print(f"  {'Class':<22} {'Soft-L1':>10} {'MAD Score':>12}  Flag")
    print("-" * 68)
    for c in range(C):
        flag_str = "  *** BACKDOOR ***" if results["flags"][c] else ""
        print(
            f"  {names[c]:<22} {results['l1_norms'][c]:>10.3f}"
            f" {results['anomaly_scores'][c]:>12.3f}{flag_str}"
        )
    print("=" * 68 + "\n")


# ---------------------------------------------------------------------------
# Mitigation: classify while rejecting triggered inputs (Section 5)
# ---------------------------------------------------------------------------

def classify_with_mitigation(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    masks: List[torch.Tensor],
    patterns: List[torch.Tensor],
    suspected_classes: List[int],
    threshold: float = 0.5,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict classes, rejecting inputs whose S(x) < threshold.

    Rejected inputs get prediction = -1.

    Returns:
        predictions: [N] with -1 for rejected inputs.
        scores:      [N] mitigation scores.
    """
    x = x.to(device)
    with torch.no_grad():
        predictions = model_fn(x).argmax(dim=1)
    scores = mitigation_score(
        model_fn, x, masks, patterns, suspected_classes, device=device)
    predictions[scores < threshold] = -1
    return predictions, scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="B3D: Black-box Backdoor Detection (Dong et al., ICCV 2021)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",       required=True,
                   help="TorchScript (.pt) model file.")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--image_shape", nargs=3, type=int, metavar=("C","H","W"),
                   required=True)
    p.add_argument("--data",        default=None,
                   help="Clean-image tensor .pt, shape [N,C,H,W] in [0,1].")
    p.add_argument("--synthetic",   action="store_true",
                   help="Use B3D-SS (synthesise images).")
    p.add_argument("--n_synthetic", type=int, default=100)
    p.add_argument("--k",           type=int,   default=50)
    p.add_argument("--sigma",       type=float, default=0.1)
    p.add_argument("--lr",          type=float, default=0.05)
    p.add_argument("--lam",         type=float, default=0.01)
    p.add_argument("--iters",       type=int,   default=1000)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save",        default=None)
    p.add_argument("--class_names", nargs="+",  default=None)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    args = _build_parser().parse_args()

    model = torch.jit.load(args.model, map_location=args.device)
    model.eval()
    model_fn: Callable = lambda x: model(x)  # noqa: E731

    X: Optional[torch.Tensor] = None
    if args.data:
        X = torch.load(args.data, map_location="cpu").float()
        if X.max() > 1.0:
            X = X / 255.0

    results = detect_backdoor(
        model_fn=model_fn,
        num_classes=args.num_classes,
        image_shape=tuple(args.image_shape),
        X=X,
        use_synthetic=args.synthetic,
        n_synthetic_per_class=args.n_synthetic,
        k=args.k, sigma=args.sigma, lr=args.lr,
        lam_init=args.lam,
        num_iterations=args.iters,
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.save,
    )
    print_report(results, args.class_names)


if __name__ == "__main__":
    main()
