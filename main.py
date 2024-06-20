import numpy as np
import torch.nn as nn
import torch

from miou.metrics import miou
from miou.utils.mask_loader import load_mask
from miou.utils.segmentation import GridBasedSegmentation
from scipy.stats import linregress

def pre_process_masks(mask1, mask2, scale):  # This is the function s in the paper
    preprocessor1 = GridBasedSegmentation(mask1, scale)
    s1 = preprocessor1.segmentize()
    preprocessor2 = GridBasedSegmentation(mask2, scale)
    s2 = preprocessor2.segmentize()

    return s1, s2


def custom_bce_simplified(target, predictions, reduction="mean"):
    m = nn.LogSigmoid()
    log_sigmoid_pred = m(predictions)
    loss = (1 - target) * predictions - log_sigmoid_pred
    if reduction == "sum":
        loss = torch.sum(loss)
    else:
        loss = torch.mean(loss)
    return loss


def calculate_ce(gt, prediction, scale):
    return custom_bce_simplified(gt, prediction)


def calculate_area(distances, scales):
    normalized_boxsizes = miou.normalize_boxsizes(scales)
    slope, _, _, _, _ = linregress(normalized_boxsizes, distances)
    aiou = np.abs(slope)

    area = miou.integral_trapezoidal(distances, dx=1 / (len(scales) - 1))
    return area


def iou(mask1, mask2):
    pass


def main():
    gt = load_mask("./test_images/test_segm_input_B/mask1.png")
    img2 = load_mask("./test_images/test_segm_input_B/mask2.png")
    img3 = load_mask("./test_images/test_segm_input_B/mask3.png")

    scales = pow(2, np.linspace(0, 9, num=10, dtype=int))

    distances = []
    box_counting_m1 = []
    box_counting_m2 = []
    for scale in scales:
        s_gt, s_img2 = pre_process_masks(gt, img2, scale)
        # box_counting_m1.append(np.sum(s_gt) + 1)
        # box_counting_m2.append(np.sum(s_gt) + 1)
        distance = calculate_ce(s_gt, s_img2, scale)
        distances.append(distance)

    mce = calculate_area(distances, scales)
    print(mce)

    # gt, img3 = pre_process_masks(gt, img3)
