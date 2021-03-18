import numpy as np


def encode_binary_mask(binary_mask: np.ndarray) -> dict:
    h, w = binary_mask.shape
    binary_mask = binary_mask.flatten(order="F")
    mask_len = len(binary_mask)
    counts_list = []
    pos = 0

    # RLE encoding
    counts_list.append(1)
    diffs = np.logical_xor(binary_mask[0 : mask_len - 1], binary_mask[1:mask_len])
    for diff in diffs:
        if diff:
            pos += 1
            counts_list.append(1)
        else:
            counts_list[pos] += 1

    # Encoding an array into a byte array
    # if array starts from 1. start with 0 counts for 0
    if binary_mask[0] == 1:
        counts_list = [0] + counts_list
    counts = []
    more = True
    i = 0
    for i in range(len(counts_list)):
        x = counts_list[i]
        if i > 2:
            x -= counts_list[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = x != -1 if (c & 0x10) else x != 0
            if more:
                c |= 0x20
            c += 48  # shift for byte
            counts.append(chr(c))

    return {
        "size": [h, w],
        "counts": "".join(counts).encode("utf-8"),
    }


def decode_coco_rle(rle_mask: dict) -> np.ndarray:
    binaries = rle_mask["counts"]
    resulted_mask = []
    more = True
    i = 0
    k = 0
    x = 0
    prev_x = 0
    prev_prev_x = 0
    while k < len(binaries):
        prev_prev_x = prev_x
        prev_x = x
        x = 0
        m = 0
        more = True
        while more:
            c = binaries[k] - 48  # shift for byte
            x |= (c & 0x1F) << 5 * m  # number of values (0 or q)
            more = c & 0x20
            k += 1
            m += 1
            if not more and (c & 0x10):
                x |= -1 << 5 * m
        if i > 2:
            x += prev_prev_x  # add previous counts
        i += 1
        value = not i % 2  # alternation of 0 and 1
        resulted_mask += [value] * x  # create a sequence of values
    return np.array(resulted_mask, dtype=np.uint8).reshape(rle_mask["size"], order="F")
