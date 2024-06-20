"""
- Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)
"""

from miou.metrics import metric
from miou.edge_detection import sobel
import numpy as np
import copy


class PrecisionRecall(metric.IMetric):
    """
    This implements the Precision and Recall, as a metric for verification of segmentations.
    This was introduced in Estrada et al., 2009.
    """
    def __init__(self, edge_only=False):
        """
        :param edge_only: the default value is `False'. This is to be comparable with IoU. However,
        it was originally introduced to be computed on the boundaries.
        """
        self.edge_only = edge_only
        self.pr = dict()
        self.fscore = 0

    def measure(self, mask1: np.ndarray, mask2: np.ndarray, *args, **kwargs):
        """
        calculates precision and recall, by comparing `mask1` (reference) and `mask2` (target),
        as follows::

            precision = |intersect(mask1, mask2)| / |mask2|

            recall    = |intersect(mask1, mask2)| / |mask1|

        where |.| is the number of pixels in that region.

        :param mask1: mask segmentation of the reference (ground-truth) object.
        :param mask2: mask segmentation of the target (detected) object.
        :param args:
        :param kwargs:
        :return: a dictionary of the calculated precision and recall, with the keys
        'Precision' and 'Recall', respectively.
        """
        self.pr = dict()
        m1 = copy.deepcopy(mask1)
        m2 = copy.deepcopy(mask2)

        if self.edge_only:
            m1 = sobel.get_edges(m1)
            m2 = sobel.get_edges(m2)

        intersect = np.logical_and(m1, m2)
        intersect_count = np.sum(intersect)
        m1_count = np.sum(m1)  # total number of pixels of m1 (reference)
        m2_count = np.sum(m2)  # total number of pixels of m2 (target)
        self.pr = {'precision': (intersect_count / m2_count),
                   'recall': (intersect_count / m1_count)}
        self.fscore = self.compute_fscore()
        return self.pr

    def compute_fscore(self):
        """
        compute the f1-score as the harmonic mean of precision and recall.
        """
        num = self.pr['precision'] * self.pr['recall']
        den = self.pr['precision'] + self.pr['recall']
        return (2 * num) / den
