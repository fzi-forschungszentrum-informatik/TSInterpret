from functools import partial

import numpy as np
from sklearn.metrics import pairwise_distances

from TSInterpret.InterpretabilityModels.leftist.learning_process.neighbors_generator.neighbors_generator import (
    NeighborsGenerator,
)
from TSInterpret.InterpretabilityModels.leftist.learning_process.utils_learning_process import (
    reconstruct,
)
from TSInterpret.InterpretabilityModels.leftist.neighbors import Neighbors

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class LIMENeighborsGenerator(NeighborsGenerator):
    """
    Module to generate neighbors as in LIME.

    Attributes:
        random_state (int): random state that will be used to generate number. If None, the random state will be
                initialized using the internal numpy seed.
        kernel_width (float): kernel width for the exponential kernel.
        kernel (python function): similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to an exponential kernel.
        distance_metric (string): distance metric use to compute kernel weights.

    """

    def __init__(
        self, random_state, kernel_width=0.25, kernel=None, distance_metric=None
    ):
        """
        Must inherit Neighbors_Generator class.

        """
        NeighborsGenerator.__init__(self)
        self.random_state = random_state
        self.kernel_width = float(kernel_width)
        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.kernel = partial(kernel, kernel_width=kernel_width)
        if distance_metric is None:
            self.distance_metric = "LIMEcosine"
        else:
            self.distance_metric = distance_metric

    def generate(self, nb_features, nb_neighbors, transform):
        """
        Generate neighbors as in LIME.

        Parameters:
            nb_features (int): number of features in the interpretable binary space.
            nb_neighbors (int): numbers of neighbors to draw in the interpretable binary space.
            transform (Transform): the transform function.

        Returns:
            neighbors (Neighbors): the neighbors.
        """
        # initialize the neighbors
        neighbors = Neighbors()
        # neighbors masks generation (simplified representation of the neighbors in the interpretable binary space)
        neighbors.masks = self.random_state.randint(
            0, 2, nb_neighbors * nb_features
        ).reshape((nb_neighbors, nb_features))
        # set the first neighbor mask as the simplified representation of the instance to explain
        neighbors.masks[0, :] = 1

        #  build neighbors values in the original space of the instance to explain
        # print(neighbors.masks)
        # print('!!!!!!!!!!!!!!!!!!')
        # print(type(neighbors.masks))
        neighbors.values = reconstruct(neighbors, transform)
        # compute the distances between the neighbors mask
        distances = self._compute_distances(neighbors)
        # compute the kernel weight of each neighbor
        neighbors = self._compute_kernel_weights(neighbors, distances)
        return neighbors

    def _compute_distances(self, neighbors):
        """
        Compute the pairwise distances between the neighbors.

        Parameters:
            neighbors (Neighbors): the neighbors.

        Returns:
            distances (np.ndarray): the distances between each neighbors.

        """
        if self.distance_metric == "LIMEcosine":
            distances = pairwise_distances(
                neighbors.masks, neighbors.masks[0].reshape(1, -1), metric="cosine"
            ).ravel()
        else:
            distances = pairwise_distances(
                neighbors.values,
                neighbors.values[0].reshape(1, -1),
                metric=self.distance_metric,
            ).ravel()

        return distances

    def _compute_kernel_weights(self, neighbors, distances):
        """
        Compute the kernel weight of each neighbor.

        Parameters:
            neighbors (Neighbors): the neighbors.
            distances (np.ndarray): the distances between each neighbors.

        Returns:
            neighbors (Neighbors): the neighbors with their kernel weight updated
        """
        neighbors.kernel_weights = self.kernel(distances)
        return neighbors
