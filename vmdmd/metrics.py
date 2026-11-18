"""Global confusion counts; DSC, precision and recall share the same population."""
import torch


class BinaryMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, prediction, target):
        p, y = prediction.bool(), target.bool()
        self.tp += int((p & y).sum())
        self.fp += int((p & ~y).sum())
        self.fn += int((~p & y).sum())
        self.tn += int((~p & ~y).sum())

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        def divide(n, d, empty=0.):
            return n/d if d else empty
        return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                    dice=divide(2*tp,2*tp+fp+fn,1.), iou=divide(tp,tp+fp+fn,1.),
                    precision=divide(tp,tp+fp), recall=divide(tp,tp+fn), sensitivity=divide(tp,tp+fn),
                    specificity=divide(tn,tn+fp), accuracy=divide(tp+tn,tp+fp+fn+tn))


BOUNDARY_PROTOCOL = {
    'definition': 'inner boundary: mask minus 3x3 erosion, outside image is background',
    'distance': 'Euclidean pixels at original mask resolution',
    'tolerance': 'max(1, ceil(0.0025 * hypot(height, width))) pixels',
    'aggregation': 'arithmetic mean of per-image boundary F1',
    'empty_masks': 'both empty: 1; exactly one empty: 0',
}


def boundary_metrics(prediction, target):
    """One binary image, symmetric boundary matching with fixed tolerance."""
    import math
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt
    p = prediction.detach().cpu().numpy().astype(bool)
    y = target.detach().cpu().numpy().astype(bool)
    if p.ndim > 2:
        if any(n != 1 for n in p.shape[:-2]):
            raise ValueError('Boundary metrics require one image and one channel')
        p, y = p.reshape(p.shape[-2:]), y.reshape(y.shape[-2:])
    if p.ndim != 2 or p.shape != y.shape:
        raise ValueError('Expected matching two-dimensional masks')
    radius = max(1, math.ceil(.0025 * math.hypot(*p.shape)))
    structure = np.ones((3,3), dtype=bool)
    bp = p & ~binary_erosion(p, structure=structure, border_value=0)
    by = y & ~binary_erosion(y, structure=structure, border_value=0)
    if not bp.any() or not by.any():
        precision = recall = float(not bp.any() and not by.any())
    else:
        precision = float((distance_transform_edt(~by)[bp] <= radius).mean())
        recall = float((distance_transform_edt(~bp)[by] <= radius).mean())
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return dict(boundary_f1=f1, boundary_precision=precision,
                boundary_recall=recall, boundary_tolerance_px=radius)
