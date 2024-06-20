"""
- Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)
"""

import abc
import numpy as np


class IMetric(abc.ABC):
    @abc.abstractmethod
    def measure(self, mask1: np.ndarray, mask2: np.ndarray, *args, **kwargs):
        pass
