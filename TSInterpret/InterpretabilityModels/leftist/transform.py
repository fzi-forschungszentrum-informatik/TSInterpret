from abc import ABC, abstractmethod

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class Transform(ABC):
    """
    Abstract class of the method to create neighbors representation in explained instance data space.

    Attributes:
        explained_instance (Object): instance to explain.
    """

    @abstractmethod
    def __init__(self, explained_instance):
        """
        Abstract __init__ must be called by concrete class.
        """
        self.explained_instance = explained_instance

    @abstractmethod
    def apply(self, neighbor_mask):
        """
        Transform a neighbor into the explained instance data space from its interpretable features representation (mask).

        Parameters:
            neighbor_mask (np.ndarray): interpretable features representation of the neighbor (mask).

        Returns:
            neighbor_values (Object): representation of the neighbor in the data space of time series to explain.
        """
        pass
