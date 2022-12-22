from abc import ABC, abstractmethod

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class NeighborsGenerator(ABC):
    """
    Abstract class for the neighbors generator.
    """

    @abstractmethod
    def __init__(self):
        """
        Abstract __init__ must be called by concrete class.
        """
        pass

    @abstractmethod
    def generate(self, nb_features, nb_neighbors, transform):
        """
        Generate neighbors and the kernel weights of the neighbors around the explained instance.

        Parameters:
            nb_features (int): number of features in the interpretable binary space.
            nb_neighbors (int): numbers of neighbors to draw in the interpretable binary space.
            transform (Transform): the transform function.

        Returns:
            neighbors (Neighbors): the neighbors.
        """
        pass
