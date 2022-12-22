import numpy as np

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class Neighbors:
    """
    Object that contains all the information related to the neighbors generated around the instance to explain.

    Attributes:
        masks (np.ndarray): binary simplified representation of the neighbors.
        values (np.ndarray): representation of the neighbors in the original space of the interpretable_transformation.
        proba_labels (np.ndarray): prediction probability of the neighbors given by the model to explain.
        kernel_weights (np.ndarray): kernel weights of the neighbors used to learn the explanation model.

    """

    def __init__(self, masks=None, kernel_weights=None, values=None, proba_labels=None):
        if masks is None:
            self.masks = None
        elif isinstance(masks, np.ndarray) and (masks.ndim == 2):
            self.masks = masks
        else:
            raise TypeError(
                "neighbors masks must be np.ndarray with two dimensions (a matrix)"
            )

        if kernel_weights is None:
            self.kernel_weights = None
        elif isinstance(kernel_weights, np.ndarray) and (kernel_weights.ndim == 1):
            self.kernel_weights = kernel_weights
        else:
            raise TypeError(
                "kernel weights must be np.ndarray with one dimensions (an array)"
            )

        if values is None:
            self.values = None
        elif isinstance(values, np.ndarray) and (values.ndim == 2):
            self.values = values
        else:
            raise TypeError(
                "neighbors values must be np.ndarray with two dimensions (a matrix)"
            )

        if proba_labels is None:
            self.proba_labels = None
        elif isinstance(proba_labels, np.ndarray) and (proba_labels.ndim == 2):
            self.proba_labels = proba_labels
        else:
            raise TypeError(
                "neighbors probability labels must be np.ndarray with two dimensions (a matrix)"
            )

    def __str__(self):
        return "masks : \n {} \n kernel_weights : \n {} \n values : \n {} \n proba_labels : \n {} \n ".format(
            self.masks, self.kernel_weights, self.values, self.proba_labels
        )
