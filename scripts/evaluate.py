"""Evaluate nnU-Net predictions on the held-out test set (WT / TC / ET Dice and HD95, in mm).

  python scripts/evaluate.py --pred preds_test --gt $nnUNet_raw/Dataset501_BrainTumour/labelsTs --out results

Predictions from nnUNetv2_predict are integer label maps (0 bg, 1 edema, 2 non-enhancing, 3 enhancing),
also when the model was trained region-based, so the same metric code applies.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from metrics import region_metrics

COLORS = np.array([[0, 0, 0], [0, 0.6, 1.0], [1.0, 0.8, 0], [1.0, 0.1, 0.1]])


def overlay(gray, label):
    g = np.clip((gray - gray.min()) / (np.ptp(gray) + 1e-8), 0, 1)
    rgb = np.stack([g] * 3, -1)
    m = label > 0
    rgb[m] = 0.55 * rgb[m] + 0.45 * COLORS[label[m]]
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="folder of predicted .nii.gz label maps")
    ap.add_argument("--gt", required=True, help="folder of ground-truth labels (labelsTs)")
    ap.add_argument("--images", default=None, help="imagesTs folder (enables qualitative figures)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-figs", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    preds = sorted(Path(args.pred).glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"No predictions found in {args.pred}")
    for i, p in enumerate(preds, 1):
        case = p.name.replace(".nii.gz", "")
        pn = nib.load(str(p))
        gn = nib.load(str(Path(args.gt) / p.name))
        pred = np.asarray(pn.dataobj).astype(np.uint8)
        gt = np.asarray(gn.dataobj).astype(np.uint8)
        if pred.shape != gt.shape:
            raise SystemExit(f"Shape mismatch for {case}: {pred.shape} vs {gt.shape}")
        spacing = tuple(float(s) for s in gn.header.get_zooms()[:3])
        m = region_metrics(pred, gt, spacing)
        m["case"] = case
        rows.append(m)
        print(f"[{i}/{len(preds)}] {case} " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k.startswith("dice")))

        img_path = Path(args.images) / f"{case}_0000.nii.gz" if args.images else None
        if img_path is not None and img_path.exists() and i <= args.n_figs and (gt > 0).any():
            img = np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)
            z = int(np.argmax((gt > 0).sum(axis=(0, 1))))
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            fl = np.rot90(img[:, :, z])
            ax[0].imshow(fl, cmap="gray"); ax[0].set_title("FLAIR")
            ax[1].imshow(overlay(fl, np.rot90(gt[:, :, z]))); ax[1].set_title("Ground truth")
            ax[2].imshow(overlay(fl, np.rot90(pred[:, :, z]))); ax[2].set_title("nnU-Net prediction")
            for a in ax:
                a.axis("off")
            fig.suptitle(f"{case} (axial slice {z}) WT Dice {m['dice_WT']:.3f}")
            fig.tight_layout()
            fig.savefig(out / "figures" / f"{case}.png", dpi=120)
            plt.close(fig)

    df = pd.DataFrame(rows).set_index("case")
    df.to_csv(out / "per_case_metrics.csv")
    summary = {c: {"mean": float(df[c].mean()), "std": float(df[c].std()), "median": float(df[c].median())}
               for c in df.columns}
    summary["n_cases"] = int(len(df))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\n=== Held-out test summary, n={len(df)} (mean +- std; NaN HD95 excluded) ===")
    for r in ("WT", "TC", "ET"):
        print(f"{r}: Dice {summary[f'dice_{r}']['mean']:.3f} +- {summary[f'dice_{r}']['std']:.3f} | "
              f"HD95 {summary[f'hd95_{r}']['mean']:.1f} mm")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([df[f"dice_{r}"].dropna() for r in ("WT", "TC", "ET")])
    ax.set_xticks([1, 2, 3], ["WT", "TC", "ET"])
    ax.set_ylabel("Dice")
    ax.set_title(f"Per-case Dice (n={len(df)})")
    fig.tight_layout()
    fig.savefig(out / "dice_boxplot.png", dpi=120)


if __name__ == "__main__":
    main()
