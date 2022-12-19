from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    def __init__(self, model=None, change=False, model_path="", backend="Func") -> None:
        """Initialize Base Model.
        Arguments:
            model: trained ML Model, either the model or the direct function call for returning the probability distribution.
            change bool:  True if dimension change is necessary.
            model_path str: path to trained model.
            backend str: ML framework. For frameworks other than TensorFlow (TF), Sklearn (SK) or PyTorch (PYT),
                        provide 'Func'.
        """
        self.model = model
        self.change = change
        self.model_path = model_path
        self.backend = backend

    @abstractmethod
    def load_model(self, path):
        """Loads the model provided at the given path.
        Arguments:
            path str: Path to the trained model-
         Returns:
            loaded model.

        """
        pass

    @abstractmethod
    def predict(self, item) -> List:
        """Unified prediction function.
        Arguments:
            item np.array: item to be classified
         Returns:
            an array of output scores for a classifier, and a singleton array of predicted value for a regressor.
        """
        pass
