from abc import ABC, abstractmethod

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class LearningProcess(ABC):
    """
    Abstract class for the neighbors explainer.
    """

    @abstractmethod
    def __init__(self):
        """
        Abstract __init__ must be called by concrete class.
        """
        self.kernel = None

    @abstractmethod
    def solve(self, neighbors, idx_label, explanation_size):
        """
        Build the explanation model from the neighbors.

        Parameters:
            neighbors (Neighbors): the neighbors.
            idx_label (int): index of the label to explain.
            explanation_size (int): size of the explanation (number of features to use in model explanation).

        Returns:
            explanation (Explanation): the coefficients of the explanation model.
        """
        pass
