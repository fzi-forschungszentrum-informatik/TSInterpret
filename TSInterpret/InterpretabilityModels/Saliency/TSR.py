
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
import torch

from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_PTY import Saliency_PTY
from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_TF import Saliency_TF
class TSR():
     def __new__(self, model, NumTimeSteps, NumFeatures, method='GRAD',mode='time'):
        if isinstance(model, torch.nn.Module):

            return Saliency_PTY(model, NumTimeSteps, NumFeatures, method=method,mode=mode)
        
        elif isinstance(model, tensorflow.keras.Model):

            return Saliency_TF(model, NumTimeSteps, NumFeatures, method=method,mode=mode)
        else: 
            raise NotImplementedError("Please use a TF or PYT Classification model! If the current model is a TF or PYT Model, try calling the wrappers directly (TF -> Saliency_TF, PYT -> Saliency_PYT)")
        
