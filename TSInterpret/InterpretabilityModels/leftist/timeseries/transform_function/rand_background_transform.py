import random

import numpy as np

from TSInterpret.InterpretabilityModels.leftist.transform import Transform

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class RandBackgroundTransform(Transform):
    """
    Method to create neighbors representation in explained instance data space.
    """

    def __init__(self, explained_instance):
        """
        Must inherit Transform class.

        Parameters:
            explained_instance (Instance): instance to explain.
        """
        Transform.__init__(self, explained_instance)
        self.background_dataset = None
        self.segments_interval = None

    def set_background_dataset(self, background_dataset):
        """
        Set the background dataset used by the transform function.

        Returns:
            background_dataset (np.ndarray): background dataset.
        """
        self.background_dataset = background_dataset

    def apply(self, neighbor_mask):
        """
        Transform a neighbor into the explained instance data space from its interpretable features representation (mask).

        Parameters:
            neighbor_mask (np.ndarray): representation of the neighbor in the explained instance data space.

        Returns:
            neighbor_values (np.ndarray): representation of the neighbor in the explained instance data space.
        """
        instance_to_copy = self.explained_instance.copy()
        neighbor_values = random.choice(self.background_dataset.copy().astype(float))
        idx_segments_to_replace = np.where(neighbor_mask)[0]
        segments_interval_to_replace = self.segments_interval[idx_segments_to_replace]
        for segment_interval in segments_interval_to_replace:
            neighbor_values[
                segment_interval[0] : segment_interval[1] + 1
            ] = instance_to_copy[segment_interval[0] : segment_interval[1] + 1]
        return neighbor_values
