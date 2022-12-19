import imp

import numpy as np

from TSInterpret.Models.PyTorchModel import PyTorchModel
from TSInterpret.Models.SklearnModel import SklearnModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"
from TSInterpret.InterpretabilityModels.utils import (sklearn_wrapper,
                                                      tensorflow_wrapper,
                                                      torch_wrapper)


def reconstruct(neighbors, transform):
    """
    Build the values of the neighbors in the original data space of the instance to explain.
    Store the values into neighbors value as a dictionary.

    Parameters:
        neighbors_masks (np.ndarray): masks of the neighbors.
        transform (Transform): the transform function.

    Returns:
        neighbors_values (np.ndarray): values of the neighbors in the original data space of the instance to explain.
    """
    neighbors_values = np.apply_along_axis(transform.apply, 1, neighbors.masks)

    dict_neighbors_value = {}
    for idx in range(len(neighbors_values)):
        dict_neighbors_value[idx] = neighbors_values[idx]

    neighbors.values = dict_neighbors_value

    return neighbors_values
