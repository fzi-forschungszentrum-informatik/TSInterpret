import numpy as np

from TSInterpret.InterpretabilityModels.leftist.transform import Transform

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class AllBackgroundTransform(Transform):
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

    def set_background_dataset(self, background_dataset, limit="full"):
        """
        Set the background dataset used by the transform function.

        Parameters:
            background_dataset (np.ndarray): the background dataset.
            limit (int or string): limit of time series to use in background dataset
        """
        if limit == "full":
            _background_dataset_to_use = background_dataset
        elif isinstance(limit, int) and limit > 0:
            if limit > len(background_dataset):
                _background_dataset_to_use = background_dataset
            else:
                _background_dataset_to_use = background_dataset[
                    np.random.choice(list(range(len(background_dataset))), limit)
                ]
        else:
            raise ValueError(
                'limit must be set as "full" or with a positive integer of example to draw in the background dataset'
            )

        self.background_dataset = _background_dataset_to_use

    def apply(self, neighbor_mask):
        """
        Transform a neighbor into the explained instance data space from its interpretable features representation (mask).

        Parameters:
            neighbor_mask (np.ndarray): representation of the neighbor in the explained instance data space.

        Returns:
            neighbor_representation (np.ndarray): representation of the neighbor in the explained instance data space.
        """
        if self.background_dataset is None:
            raise ValueError("background dataset must be set to be modified")

        neighbor_representation = self.background_dataset.copy().astype(float)
        idx_segments_to_replace = np.where(neighbor_mask)[0]
        segments_interval_to_replace = self.segments_interval[idx_segments_to_replace]
        for segment_interval in segments_interval_to_replace:
            if len(neighbor_representation.shape) == 1:
                neighbor_representation[
                    segment_interval[0] : segment_interval[1] + 1
                ] = self.explained_instance[
                    segment_interval[0] : segment_interval[1] + 1
                ]
            else:
                neighbor_representation[
                    :, segment_interval[0] : segment_interval[1] + 1
                ] = self.explained_instance[
                    segment_interval[0] : segment_interval[1] + 1
                ]

        return neighbor_representation
