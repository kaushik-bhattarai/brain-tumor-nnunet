"""Convert MSD Task01_BrainTumour into nnU-Net v2 raw format, with a held-out test set.

MSD stores each case as one 4D NIfTI (H, W, D, 4 modalities: FLAIR, T1w, T1gd, T2w).
nnU-Net wants one 3D file per modality: <case>_0000.nii.gz ... <case>_0003.nii.gz.

What this script does
  1. Splits the labelled cases into train / test (default 10% test, seeded, case-level).
  2. Writes imagesTr/labelsTr (train) and imagesTs/labelsTs (held-out test) under
     $nnUNet_raw/Dataset<ID>_BrainTumour/.
  3. Writes dataset.json. With --regions (default) it uses nnU-Net's region-based
     training for the BraTS regions WT / TC / ET, which is the recommended setup for BraTS-style labels.
  4. Saves split.json so the split is reproducible and documented.

Usage
  export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
  python scripts/prepare_dataset.py --msd-dir /path/to/Task01_BrainTumour --dataset-id 501
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

MODALITIES = {"0": "FLAIR", "1": "T1w", "2": "T1gd", "3": "T2w"}


def split_4d(src: Path, dst_dir: Path, case: str):
    nii = nib.load(str(src))
    data = np.asarray(nii.dataobj)
    if data.ndim != 4 or data.shape[-1] != 4:
        raise ValueError(f"{src} should be 4D with 4 modalities, got shape {data.shape}")
    for c in range(4):
        out = nib.Nifti1Image(data[..., c].astype(np.float32), nii.affine, nii.header)
        out.set_data_dtype(np.float32)
        nib.save(out, str(dst_dir / f"{case}_{c:04d}.nii.gz"))


def copy_label(src: Path, dst: Path):
    nii = nib.load(str(src))
    data = np.asarray(nii.dataobj).astype(np.uint8)
    out = nib.Nifti1Image(data, nii.affine, nii.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(dst))


def build_dataset_json(n_train: int, regions: bool):
    if regions:
        # Order matters: encompassing regions first. Do not sort keys.
        labels = {"background": 0, "whole_tumor": [1, 2, 3], "tumor_core": [2, 3], "enhancing_tumor": 3}
    else:
        labels = {"background": 0, "edema": 1, "non_enhancing_tumor": 2, "enhancing_tumor": 3}
    ds = {
        "channel_names": {k: v for k, v in MODALITIES.items()},
        "labels": labels,
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }
    if regions:
        ds["regions_class_order"] = [1, 2, 3]
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msd-dir", required=True, help="Folder containing imagesTr/ and labelsTr/ (Task01_BrainTumour)")
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--name", default="BrainTumour")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="only use N cases (for smoke tests)")
    ap.add_argument("--no-regions", action="store_true", help="plain label-based training instead of WT/TC/ET regions")
    args = ap.parse_args()

    raw = os.environ.get("nnUNet_raw")
    if not raw:
        raise SystemExit("Set the nnUNet_raw environment variable first (see README).")
    msd = Path(args.msd_dir)
    out = Path(raw) / f"Dataset{args.dataset_id:03d}_{args.name}"
    if out.exists():
        raise SystemExit(f"{out} already exists; delete it first if you want to rebuild.")
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out / sub).mkdir(parents=True)

    cases = sorted(p.name.replace(".nii.gz", "") for p in (msd / "imagesTr").glob("*.nii.gz") if not p.name.startswith("."))
    if args.limit:
        cases = cases[: args.limit]
    if len(cases) < 3:
        raise SystemExit(f"Found only {len(cases)} cases in {msd / 'imagesTr'}")
    rng = random.Random(args.seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(args.test_frac * len(cases)))
    test, train = sorted(shuffled[:n_test]), sorted(shuffled[n_test:])
    print(f"{len(cases)} labelled cases -> {len(train)} train, {len(test)} held-out test")

    for name, group in (("Tr", train), ("Ts", test)):
        for i, case in enumerate(group, 1):
            split_4d(msd / "imagesTr" / f"{case}.nii.gz", out / f"images{name}", case)
            copy_label(msd / "labelsTr" / f"{case}.nii.gz", out / f"labels{name}" / f"{case}.nii.gz")
            if i % 50 == 0:
                print(f"  {name}: {i}/{len(group)}")

    with open(out / "dataset.json", "w") as f:
        json.dump(build_dataset_json(len(train), not args.no_regions), f, indent=2, sort_keys=False)
    with open(out / "split.json", "w") as f:
        json.dump({"seed": args.seed, "train": train, "test": test}, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
