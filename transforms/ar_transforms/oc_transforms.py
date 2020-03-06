import numpy as np
import torch
# from skimage.color import rgb2yuv
import cv2
from fast_slic.avx2 import SlicAvx2 as Slic
from skimage.segmentation import slic as sk_slic


def run_slic_pt(img_batch, n_seg=200, compact=10, rd_select=(8, 16), fast=True):  # Nx1xHxW
    """

    :param img: Nx3xHxW 0~1 float32
    :param n_seg:
    :param compact:
    :return: Nx1xHxW float32
    """
    B = img_batch.size(0)
    dtype = img_batch.type()
    img_batch = np.split(
        img_batch.detach().cpu().numpy().transpose([0, 2, 3, 1]), B, axis=0)
    out = []
    if fast:
        fast_slic = Slic(num_components=n_seg, compactness=compact, min_size_factor=0.8)
    for img in img_batch:
        img = np.copy((img * 255).squeeze(0).astype(np.uint8), order='C')
        if fast:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            seg = fast_slic.iterate(img)
        else:
            seg = sk_slic(img, n_segments=200, compactness=10)

        if rd_select is not None:
            n_select = np.random.randint(rd_select[0], rd_select[1])
            select_list = np.random.choice(range(0, np.max(seg) + 1), n_select,
                                           replace=False)

            seg = np.bitwise_or.reduce([seg == seg_id for seg_id in select_list])
        out.append(seg)
    x_out = torch.tensor(np.stack(out)).type(dtype).unsqueeze(1)
    return x_out


def random_crop(img, flow, occ_mask, crop_sz):
    """

    :param img: Nx6xHxW
    :param flows: n * [Nx2xHxW]
    :param occ_masks: n * [Nx1xHxW]
    :param crop_sz:
    :return:
    """
    _, _, h, w = img.size()
    c_h, c_w = crop_sz

    if c_h == h and c_w == w:
        return img, flow, occ_mask

    x1 = np.random.randint(0, w - c_w)
    y1 = np.random.randint(0, h - c_h)
    img = img[:, :, y1:y1 + c_h, x1: x1 + c_w]
    flow = flow[:, :, y1:y1 + c_h, x1: x1 + c_w]
    occ_mask = occ_mask[:, :, y1:y1 + c_h, x1: x1 + c_w]

    return img, flow, occ_mask
