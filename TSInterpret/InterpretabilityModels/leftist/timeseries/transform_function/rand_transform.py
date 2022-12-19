import random

import numpy as np

from TSInterpret.InterpretabilityModels.leftist.transform import Transform

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class RandTransform(Transform):
    """
    Method to create neighbors representation in explained instance data space.
    """

    def __init__(self, ts_explained_instance):
        """
        Must inherit Transform class.

        Parameters:
            ts_explained_instance (Instance): instance to explain.
        """
        Transform.__init__(self, ts_explained_instance)
        self.segments_interval = None

    def apply(self, neighbor_mask):
        """
        Transform a neighbor into the explained instance data space from its interpretable features representation (mask).

        Parameters:
            neighbor_mask (np.ndarray): representation of the neighbor in the explained instance data space.

        Returns:
            neighbor_values (np.ndarray): representation of the neighbor in the explained instance data space.
        """
        neighbor_values = self.explained_instance.copy()
        v_min = min(neighbor_values)
        v_max = max(neighbor_values)
        idx_segments_to_replace = np.where(np.array(neighbor_mask) == 0.0)
        segments_interval_to_replace = self.segments_interval[idx_segments_to_replace]
        for segment_interval in segments_interval_to_replace:
            neighbor_values[segment_interval[0] : segment_interval[1] + 1] = [
                random.uniform(v_min, v_max)
                for i in range(segment_interval[1] - segment_interval[0] + 1)
            ]
        return neighbor_values
