import json
import os

import cv2
import numpy as np

from modelplace_api._rle_mask import decode_coco_rle, encode_binary_mask

binary_mask = np.zeros((4000, 3000))
binary_mask = np.asfortranarray(
    cv2.circle(binary_mask, (2000, 1500), 1000, (255, 255, 255), -1) // 255,
).astype(np.uint8)


with open("decoded_binary_mask.json", "r") as f:
    encoded_binary_mask = json.load(f)
encoded_binary_mask["counts"] = encoded_binary_mask["counts"].encode("utf-8")


def test_encode():
    python_encoded_mask = encode_binary_mask(binary_mask)
    assert python_encoded_mask == encoded_binary_mask


def test_decode():
    python_decoded_mask = decode_coco_rle(encoded_binary_mask)
    assert np.all(python_decoded_mask == binary_mask)
