from abc import ABC, abstractmethod


class InterpretabilityBase(ABC):
    """
    Abstract class to implement custom interpretability methods
    ----------
    mlmodel:

    Methods
    -------
    explain:
        explains a instance
    Returns
    -------
    None
    """

    def __init__(self, mlmodel, mode):
        self.model = mlmodel
        self.mode = mode

    @abstractmethod
    def explain(self):
        """
        Explains instance or model.
        Parameters
        ----------
        instance: np.array
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).
        Returns
        -------
        pd.DataFrame
            Encoded and normalised counterfactual examples.
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Plots expalantion on the explained Sample.

        Parameters
        ----------
        instance: np.array
            timeseries instance in two-dimensional shape (m, n).
        exp: expalantaion

        Returns
        -------
        matplotlib.pyplot.Figure
            Visualization of Explanation.
        """
        pass
