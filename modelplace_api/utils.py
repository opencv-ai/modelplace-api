import os
from typing import Any

import numpy as np
from loguru import logger

try:
    from pycocotools import mask

    encode_binary_mask = mask.encode
    decode_coco_rle = mask.decode
except ImportError:
    logger.warning(
        "The 'pycocotools' package wasn't found. Slow encoding and decoding are used for the RLE mask.",
    )

    from ._rle_mask import encode_binary_mask, decode_coco_rle


def is_numpy_array_equal(result: np.ndarray, gt: np.ndarray, error: float) -> bool:
    '''
    Applies element-wise comparison of two ndarrays and defines
    whether the ratio of matched elements is eligible
    '''
    if result.size != gt.size:
        raise RuntimeError('"result" and "gt" arrays must have equal size')
    elements_num = result.size
    equal_elements_num = np.equal(result, gt).sum()
    matching_ratio = equal_elements_num / elements_num
    return matching_ratio >= 1 - error


def is_equal(result: Any, gt: Any, error: float = 0.001) -> bool:
    if type(result) != type(gt):
        raise TypeError

    ret = True
    if isinstance(result, dict):
        for key in result:
            ret = ret and is_equal(result[key], gt[key], error)
    elif isinstance(result, list):
        if not result:
            ret = ret and not bool(len(gt))
        else:
            for r, g in zip(result, gt):
                ret = ret and is_equal(r, g, error)
    elif isinstance(result, str):
        ret = ret and result == gt
    elif isinstance(result, bytes):
        result = result.decode("utf-8")
        gt = gt.decode("utf-8")
        ret = ret and result == gt
    elif isinstance(result, np.ndarray):
        ret = ret and is_numpy_array_equal(result, gt, error)
    else:
        ret = ret and np.isclose(result, gt, rtol=error)
    return ret


def prepare_mask(result_mask: np.ndarray) -> dict:
    masks = {
        "binary": [],
        "classes": [],
    }
    for unique in np.unique(result_mask):
        binary_mask = np.zeros(shape=result_mask.shape, dtype=np.uint8)
        binary_mask[result_mask == unique] = 1
        masks["binary"].append(encode_binary_mask(np.asfortranarray(binary_mask)))
        masks["classes"].append(int(unique))
    return masks
