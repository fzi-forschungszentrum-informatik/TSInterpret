import copy
import itertools

import numpy as np
from scipy.special import binom

from TSInterpret.InterpretabilityModels.leftist.learning_process.neighbors_generator.neighbors_generator import (
    NeighborsGenerator,
)
from TSInterpret.InterpretabilityModels.leftist.learning_process.utils_learning_process import (
    reconstruct,
)
from TSInterpret.InterpretabilityModels.leftist.neighbors import Neighbors

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class SHAPNeighborsGenerator(NeighborsGenerator):
    """
    Module to generate neighbors as in SHAP.

    Attributes:
        num_full_subsets (int): number of subsets that can be filled by remaining neighbors to draw.
        num_samples_left (int): number of remaining neighbors to draw.
        nsamplesAdded (int): number of drawn neighbors.
        num_subset_sizes (int): number of subset size.
        num_paired_subset_sizes (int): number of subset size which has a complement.
        weights_vectors (int): kernel weights associated with each subset size
        neighbors (int): the generated neighbors
    """

    def __init__(self):
        """
        Must inherit Neighbors_Generator class.

        """
        NeighborsGenerator.__init__(self)
        self.num_full_subsets = 0
        self.num_samples_left = None
        self.nsamplesAdded = 0
        self.num_subset_sizes = None
        self.num_paired_subset_sizes = None
        self.weight_vector = None
        self.neighbors = None

    def generate(self, nb_features, nb_neighbors, transform):
        """
        Generate neighbors as in SHAP.

        Parameters:
            nb_features (int): number of features in the interpretable binary space.
            nb_neighbors (int): numbers of neighbors to draw in the interpretable binary space.
            transform (Transform): the transform function.

        Returns:
            neighbors (Neighbors): the neighbors.
        """
        # initialize attributes
        self._allocate(nb_features, nb_neighbors)

        # enumerate all the possible subset according the number of features and the number of neighbors
        self._enumerate_full_subset(nb_features)

        # if there is still neighbors to draw we pick randomly in the remaining subsets
        if self.num_full_subsets != self.num_subset_sizes:
            self._pick_in_random_subset(nb_features)

        #  build neighbors values in the original space of the instance to explain
        self.neighbors.values = reconstruct(self.neighbors, transform)

        return self.neighbors

    def _allocate(self, nb_features, nb_neighbors):
        """
        Update the attributes according the number of features and the number of neighbors.

        Parameters:
            nb_features (int): number of features in the interpretable binary space.
            nb_neighbors (int): numbers of neighbors to draw in the interpretable binary space.
        """
        self.num_samples_left = nb_neighbors
        if self.num_samples_left > 2**nb_features - 2:
            self.num_samples_left = 2**nb_features - 2
        self.neighbors = Neighbors(
            masks=np.zeros((self.num_samples_left, nb_features)),
            kernel_weights=np.zeros(self.num_samples_left),
        )
        self.num_subset_sizes = np.int(np.ceil((nb_features - 1) / 2.0))
        self.num_paired_subset_sizes = np.int(np.floor((nb_features - 1) / 2.0))
        self.weight_vector = np.array(
            [
                (nb_features - 1.0) / (i * (nb_features - i))
                for i in range(1, self.num_subset_sizes + 1)
            ]
        )
        self.weight_vector[: self.num_paired_subset_sizes] *= 2
        self.weight_vector /= np.sum(self.weight_vector)

    def _enumerate_full_subset(self, nb_features):
        """
        Enumerate the subset that can be completely filled with the remaining neighbors to draw.

        Parameters:
            nb_features (int): number of features in the interpretable binary space.

        Returns:
            neighbors (Neighbors): add mask and kernel weight of the neighbors that completely filled subset.
        """
        # weigth vector used to check if we fill subset
        remaining_weight_vector = copy.copy(self.weight_vector)

        # determine how many subsets (and their complements) are of the size 1
        subset_size = 1
        nsubsets = self._compute_nb_subset(nb_features, subset_size)

        # check if we can completely fill the subset size with the remaining neighbors
        while (subset_size < self.num_subset_sizes + 1) and (
            self.num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets
            >= 1.0 - 1e-8
        ):

            # update the counter of full subset and the remaining neighbors to draw
            self.num_full_subsets += 1
            self.num_samples_left -= nsubsets

            # rescale what's left of the remaining weight vector to sum to 1
            if remaining_weight_vector[subset_size - 1] < 1.0:
                remaining_weight_vector /= 1 - remaining_weight_vector[subset_size - 1]

            # generate mask and kernel weight of all the neighbors in this subset size
            self._generate_neighbors_for_complete_subset(nb_features, subset_size)

            # determine how many subsets (and their complements) are of the next size
            subset_size += 1
            nsubsets = self._compute_nb_subset(nb_features, subset_size)

    def _pick_in_random_subset(self, nb_features):
        """
        Enumerate the subset that can be completely filled with the remaining neighbors to draw.

        Parameters:
            nb_features (int): number of features in the interpretable binary space.

        Returns:
            neighbors (Neighbors): add mask and kernel weight of the remaining neighbors to draw that didn't filled subset.
        """
        nfixed_samples = self.nsamplesAdded
        remaining_weight_vector = copy.copy(self.weight_vector)
        remaining_weight_vector[
            : self.num_paired_subset_sizes
        ] /= 2  # because we draw two samples each below
        remaining_weight_vector = remaining_weight_vector[self.num_full_subsets :]
        remaining_weight_vector /= np.sum(remaining_weight_vector)
        used_masks = {}
        neighbor_mask = np.zeros(nb_features)
        while self.num_samples_left > 0:
            neighbor_mask.fill(0.0)
            ind = np.random.choice(
                len(remaining_weight_vector), 1, p=remaining_weight_vector
            )[0]
            subset_size = ind + self.num_full_subsets + 1
            neighbor_mask[np.random.permutation(nb_features)[:subset_size]] = 1.0

            # only add the sample if we have not seen it before, otherwise just
            # increment a previous sample's weight
            mask_tuple = tuple(neighbor_mask)
            new_sample = False
            if mask_tuple not in used_masks:
                new_sample = True
                used_masks[mask_tuple] = self.nsamplesAdded
                self.num_samples_left -= 1
                self._add_sample(self.nsamplesAdded, neighbor_mask, 1.0)
            else:
                self.neighbors.kernel_weights[used_masks[mask_tuple]] += 1.0

            # add the compliment sample
            if (
                self.num_samples_left > 0
                and subset_size <= self.num_paired_subset_sizes
            ):

                neighbor_mask[:] = np.abs(neighbor_mask - 1)
                # only add the sample if we have not seen it before, otherwise just
                # increment a previous sample's weight
                if new_sample:
                    self.num_samples_left -= 1
                    self._add_sample(self.nsamplesAdded, neighbor_mask, 1.0)
                else:
                    # we know the compliment sample is the next one after the original sample, so + 1
                    self.neighbors.kernel_weights[used_masks[mask_tuple]] += 1.0

        # normalize the kernel weights for the random samples to equal the weight left after
        # the fixed enumerated samples have been already counted
        weight_left = np.sum(self.weight_vector[self.num_full_subsets :])
        self.neighbors.kernel_weights[nfixed_samples:] *= (
            weight_left / self.neighbors.kernel_weights[nfixed_samples:].sum()
        )

    def _compute_nb_subset(self, nb_features, subset_size):
        """
        Compute the number of subset for subset size provided and the number of features.

        Parameters:
            nb_features (int): number of features int the interpretable binary space.
            subset_size (int): size of the subset

        Returns:
            nsubsets (int): number of subset
        """
        nsubsets = binom(nb_features, subset_size)
        if subset_size <= self.num_paired_subset_sizes:
            nsubsets *= 2
        return nsubsets

    def _add_sample(self, idx, mask, kernel_weight):
        """
        Add the mask (interpretable representation) and the kernel weight of a drawn neighbor.

        Parameters:
            idx (int): the index of the drawn neighbor.
            mask (np.ndarray): the mask of the drawn neighbor.
            kernel_weight (float): the kernel weight of the drawn neighbor.
        """
        self.neighbors.masks[idx, :] = mask
        self.neighbors.kernel_weights[idx] = kernel_weight
        self.nsamplesAdded += 1

    def _generate_neighbors_for_complete_subset(self, nb_features, subset_size):
        """
        Generate the mask (interpretable representation) and the kernel weight for the neighbors of a complete subset size.

        Parameters:
            nb_features (int): number of features int the interpretable binary space.
            subset_size (int): size of subset to fill

        Returns:
            neighbors (Neighbors): add mask and kernel weight of the neighbors that completely filled one subset size.
        """
        group_inds = np.arange(nb_features, dtype="int64")
        neighbor_mask = np.zeros(nb_features)

        # compute the kernel weight for the neighbor of the current subset size
        neighbor_kernel_weight = self.weight_vector[subset_size - 1] / binom(
            nb_features, subset_size
        )
        if subset_size <= self.num_paired_subset_sizes:
            neighbor_kernel_weight /= 2.0

        # generate the mask and add the kernel weight for the neighbor in the current subset size
        for inds in itertools.combinations(group_inds, subset_size):
            neighbor_mask[:] = 0.0
            neighbor_mask[np.array(inds, dtype="int64")] = 1.0
            self._add_sample(self.nsamplesAdded, neighbor_mask, neighbor_kernel_weight)
            if subset_size <= self.num_paired_subset_sizes:
                neighbor_mask[:] = np.abs(neighbor_mask - 1)
                self._add_sample(
                    self.nsamplesAdded, neighbor_mask, neighbor_kernel_weight
                )
