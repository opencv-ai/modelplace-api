from typing import Any

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
    else:
        ret = ret and np.isclose(result, gt, rtol=error)
    return ret
