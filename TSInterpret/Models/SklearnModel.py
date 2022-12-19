from TSInterpret.Models.base_model import BaseModel


class SklearnModel(BaseModel):
    def __init__(self, model, change=False) -> None:
        """Wrapper for Sklearn Models that unifiy the prediction function for a classifier.
        Arguments:
            model : Trained Sklearn Model.
            change bool: if swapping of dimension is necessary = True
        """

        super().__init__(model, change, model_path="", backend="SK")

    def predict(self, item):
        """Unified prediction function.
        Arguments:
            item np.array: item to be classified.
         Returns:
            an array of output scores for a classifier.
        """
        if self.change:
            item = item.reshape(item.shape[0], item.shape[2], item.shape[1])
        out = self.model.predict_proba(item)
        return out

    def load_model(self, path):
        return super().load_model(path)
