import os

import cv2
import numpy as np
from pycocotools import mask

from modelplace_api._rle_mask import decode_coco_rle, encode_binary_mask

binary_mask = np.zeros((4000, 3000))
binary_mask = np.asfortranarray(
    cv2.circle(binary_mask, (2000, 1500), 1000, (255, 255, 255), -1) // 255,
).astype(np.uint8)


def test_encode():
    pycocotools_encoded_mask = mask.encode(binary_mask)
    python_encoded_mask = encode_binary_mask(binary_mask)
    assert python_encoded_mask == pycocotools_encoded_mask


def test_decode():
    pycocotools_encoded_mask = mask.encode(binary_mask)
    python_decoded_mask = decode_coco_rle(pycocotools_encoded_mask)
    assert np.all(python_decoded_mask == binary_mask)
