"""BraTS-style region metrics: WT (whole tumour), TC (tumour core), ET (enhancing tumour)."""
import numpy as np
from scipy import ndimage

REGIONS = {
    "WT": lambda l: l > 0,  # edema + non-enhancing + enhancing
    "TC": lambda l: (l == 2) | (l == 3),  # non-enhancing + enhancing
    "ET": lambda l: l == 3,  # enhancing
}


def dice(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum())


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """95th-percentile Hausdorff distance in mm. NaN if either mask is empty."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d_to_gt = ndimage.distance_transform_edt(~sg, sampling=spacing)[sp]
    d_to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)[sg]
    return float(np.percentile(np.hstack([d_to_gt, d_to_pred]), 95))


def region_metrics(pred_label, gt_label, spacing=(1.0, 1.0, 1.0)):
    out = {}
    for name, fn in REGIONS.items():
        p, g = fn(pred_label), fn(gt_label)
        out[f"dice_{name}"] = dice(p, g)
        out[f"hd95_{name}"] = hd95(p, g, spacing)
    return out