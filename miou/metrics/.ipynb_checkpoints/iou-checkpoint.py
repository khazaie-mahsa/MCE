"""
- Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)
"""

from miou.metrics import metric
from miou.edge_detection import sobel
import numpy as np
import copy


class IoU(metric.IMetric):
    """
    This implements Intersection over Union, as a metric for verification of segmentations.
    This was introduced in Ge et al., 2009.
    """

    def __init__(self, edge_only=False):
        """
        Class constructor.

        :param edge_only: the default value is `False`. Usage of `True` is only experimental.
        """
        self.edge_only = edge_only

    def measure(self, mask1: np.ndarray, mask2: np.ndarray, *args, **kwargs):
        """
        calculates IoU, by comparing `mask1` (reference) and `mask2` (target), as follows::

            IoU = |intersect(mask1, mask2)| / |union(mask1, mask2)|

        where |.| is the number of pixels in that region.

        :param mask1: mask segmentation of the reference (ground-truth) object.
        :param mask2: mask segmentation of the target (detected) object.
        :param args:
        :param kwargs:
        :return: IoU.
        """
        m1 = copy.deepcopy(mask1)
        m2 = copy.deepcopy(mask2)

        if self.edge_only:
            m1 = sobel.get_edges(m1)
            m2 = sobel.get_edges(m2)

        intersect = np.logical_and(m1, m2)
        intersect_count = np.sum(intersect)
        union = np.logical_or(m1, m2)
        union_count = np.sum(union)

        return intersect_count / union_count