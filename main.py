import copy

import numpy as np
import scipy.integrate
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler

from miou.metrics import miou
from miou.utils.mask_loader import load_mask
from miou.utils.segmentation import GridBasedSegmentation
from scipy.stats import linregress
from miou.edge_detection import sobel
from scipy.integrate import trapezoid, simpson


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
    # return custom_bce_simplified(gt, prediction)
    print(prediction)
    print(gt)
    loss = nn.BCEWithLogitsLoss()
    # loss = nn.BCELoss()
    # loss = nn.CrossEntropyLoss()
    l = loss(prediction, gt)
    print(l)
    print("-------------------------------------------------------------")
    return l


def calculate_area(distances, scales):
    b = copy.deepcopy(scales).reshape((-1, 1))
    normalized_scales = MinMaxScaler().fit_transform(b)
    normalized_scales = normalized_scales.reshape(-1)
    print(f'distances after conversion:{distances}')
    area = simpson(distances, normalized_scales)
    return area


def trapezoidal_rule(x, y):
    """
    Calculate the area under the curve using the trapezoidal rule.

    Parameters:
    x (list or numpy array): The x coordinates of the points.
    y (list or numpy array): The y coordinates of the points.

    Returns:
    float: The area under the curve.
    """
    area = 0.0
    n = len(x)

    for i in range(1, n):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0

    return area

def iou(mask1, mask2):
    pass


def main():
    gt = load_mask("./test_images/test_segm_input_B/mask1.png")
    img2 = load_mask("./test_images/test_segm_input_B/mask2.png")
    # img2 = load_mask("./test_images/test_segm_input_B/mask3.png")
    # img2 = load_mask("./test_images/test_segm_input_B/mask1.png")

    scales = np.power(2, np.linspace(0, 9, num=10, dtype=int))

    distances = []
    box_counting_m1 = []
    box_counting_m2 = []
    box_counting_distance = []
    gt = sobel.get_edges(gt)
    img2 = sobel.get_edges(img2)
    for scale in scales:
        s_gt, s_img2 = pre_process_masks(gt, img2, scale)
        box_counting_m1.append(np.sum(s_gt) + 1)
        box_counting_m2.append(np.sum(s_img2) + 1)
        # distance = calculate_ce(torch.from_numpy(s_gt.astype(float)), torch.from_numpy(s_img2.astype(float)), scale)
        # distance = calculate_ce(
        #     torch.tensor(box_counting_m1[-1], dtype=torch.float),
        #     torch.tensor(box_counting_m2[-1], dtype=torch.float),
        #     scale)
        distance = calculate_ce(
            torch.tensor(s_gt, dtype=torch.float),
            torch.tensor(s_img2, dtype=torch.float),
            scale)
        # print(f"scale: {scale}")
        # print(f"loss: {distance}")
        box_counting_distance.append(torch.sum(distance))
        print(f"box_counting_1: {box_counting_m1}")
        print(f"box_counting_2: {box_counting_m2}")
        distances.append(distance)

    miou_obj = miou.MIoU(scales, edge_only=True)
    miou_obj.measure(gt, img2)
    print(f"miou: {miou_obj.area}")

    scales = miou.normalize_boxsizes(scales)
    print(f"distances: {distances}")
    mce = calculate_area(distances, scales)
    print(f"mce: {mce}")

    # gt, img3 = pre_process_masks(gt, img3)

main()
