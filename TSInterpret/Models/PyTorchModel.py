"""Module containing an interface to trained PyTorch model."""

import numpy as np 
from TSInterpret.Models.base_model import BaseModel
from typing import List
import torch

class PyTorchModel(BaseModel):
    def __init__(self,model,change = False) -> None:
        """ Wrapper for PyTorch Models that unifiy the prediction function for a classifier.
        Arguments:
            model : Trained PYT Model.
            change bool: if swapping of dimension is necessary = True 
        """
        try:
            import torch
        except ImportError:
            assert "Please install torch via pip install .[torch]"


        super().__init__(model,change, model_path='', backend='PYT')
       
    def predict(self,item)->List:
        """ Unified prediction function.
        Arguments:
            item np.array: item to be classified.
         Returns:
            an array of output scores for a classifier.
        """
        item = np.array(item.tolist(), dtype=np.float64)
        if self.change:
            item=torch.from_numpy(item.reshape(-1,item.shape[-1],item.shape[-2]))
            
        else: 
            item = torch.from_numpy(item)
        out=self.model(item.float())
        y_pred = torch.nn.functional.softmax(out).detach().numpy()
        return y_pred
    def load_model(self, path):
        return super().load_model(path)


    