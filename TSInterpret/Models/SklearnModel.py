
from Models.base_model import BaseModel


class SklearnModel(BaseModel):

    def __init__(self,model, change) -> None:
        super().__init__(model,change, model_path='', backend='Sklearn')

    def predict(self,item):
        if self.change:
            item=item.reshape(item.shape[0],item.shape[2],item.shape[1])
        out=self.model.predict_proba(item)
        return out