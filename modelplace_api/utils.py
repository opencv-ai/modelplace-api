from typing import Any
from pycocotools import mask
import numpy as np


def is_equal(result: Any, gt: Any, error: float = 0.001) -> bool:
    if type(result) != type(gt):
        raise TypeError

    ret = True
    if isinstance(result, dict):
        for key in result:
            ret = ret and is_equal(result[key], gt[key], error)
    elif isinstance(result, list):
        for r, g in zip(result, gt):
            ret = ret and is_equal(r, g, error)
    elif isinstance(result, str):
        ret = ret and result == gt
    elif isinstance(result, bytes):
        result = result.decode("utf-8")
        gt = gt.decode("utf-8")
        ret = ret and result == gt
    else:
        ret = ret and np.isclose(result, gt, rtol=error)
    return ret

def prepare_mask(result_mask):
    masks = {
        "binary": [],
        "classes": [],
    }
    for unique in np.unique(result_mask):
        binary_mask = np.zeros(shape=result_mask.shape, dtype=np.uint8)
        binary_mask[result_mask == unique] = 1
        masks["binary"].append(
            mask.encode(np.asfortranarray(binary_mask, dtype=np.uint8)),
        )
        masks["classes"].append(int(unique))
    return masks
