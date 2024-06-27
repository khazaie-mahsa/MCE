import copy

import numpy as np
import scipy.integrate
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import xticks
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


def calculate_bce_with_logits(gt, prediction, scale):
    loss = nn.BCEWithLogitsLoss()
    l = loss(prediction, gt)
    return l


def calculate_area(distances, scales):
    b = copy.deepcopy(scales).reshape((-1, 1))
    normalized_scales = MinMaxScaler().fit_transform(b)
    normalized_scales = normalized_scales.reshape(-1)
    print(f'distances after conversion:{distances}')
    area = simpson(distances, normalized_scales)
    return area


def visualize_distance(distances):
    colors = ['green', 'red', 'orange', 'blue']
    labels = ['gt-gt', 'gt-mask2', 'gt-mask3']
    scales = [*distances[0].keys()]
    for i, img_distances in enumerate(distances):
        plt.plot(scales, img_distances.values(), label=labels[i], color=colors[i])
        xticks(scales, scales)
        plt.title('Visualization of distances at each scale (not normalized)')
        plt.xlabel('scale')
        plt.ylabel('distance')
    plt.grid(axis='x', color='0.95')
    plt.legend()
    plt.show()


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
    all_distances = []
    # ------------------------example1-------------------------------
    # gt = load_mask("./test_images/test_segm_input_B/mask1.png")
    # img2 = load_mask("./test_images/test_segm_input_B/mask2.png")
    # img3 = load_mask("./test_images/test_segm_input_B/mask3.png")
    # img4_gt = load_mask("./test_images/test_segm_input_B/mask1.png")

    # ------------------------example2-------------------------------
    # gt = load_mask("./test_images/test_segm_input_Z2/mask1.png")
    # img2 = load_mask("./test_images/test_segm_input_Z2/mask2.png")
    # img3 = load_mask("./test_images/test_segm_input_Z2/mask3.png")
    # img4_gt = load_mask("./test_images/test_segm_input_Z2/mask1.png")
    # -----------------------example3--------------------------------
    gt = load_mask("./test_images/test_segm_input_Z3/mask1.png")
    img2 = load_mask("./test_images/test_segm_input_Z3/mask2.png")
    img3 = load_mask("./test_images/test_segm_input_Z3/mask3.png")
    img4_gt = load_mask("./test_images/test_segm_input_Z3/mask1.png")
    test_cases = [img4_gt, img2, img3]
    scales = np.power(2, np.linspace(0, 9, num=10, dtype=int))

    for img in test_cases:
        distances = []
        gt = sobel.get_edges(gt)
        img = sobel.get_edges(img)
        distance_dict = {}
        for scale in scales:
            s_gt, s_img2 = pre_process_masks(gt, img, scale)
            distance = calculate_bce_with_logits(
                torch.tensor(s_gt, dtype=torch.float),
                torch.tensor(s_img2, dtype=torch.float),
                scale)
            distances.append(distance)
            distance_dict[scale] = distance

        all_distances.append(distance_dict)

        miou_obj = miou.MIoU(scales, edge_only=True)
        miou_obj.measure(gt, img)
        print(f"miou: {miou_obj.area}")

        # scales = miou.normalize_boxsizes(scales)
        print(f"distances: {distances}")
        mce = calculate_area(distances, scales)
        print(f"mce: {mce}")

    # gt, img3 = pre_process_masks(gt, img3)
    visualize_distance(all_distances)


main()
