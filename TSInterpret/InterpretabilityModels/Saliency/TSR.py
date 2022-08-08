
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
import torch

from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_PTY import Saliency_PTY
from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_TF import Saliency_TF
class TSR():
    '''
    Wrapper Class for Saliency Calculation. Automatically calls the corresponding PYT or TF implementation. 
    Arguments:
        model: model to be explained
        NumTimeStep int : Number of Time Step 
        NumFetaures int : Number Features
        method str: Saliency Methode to be used
        mode str: Second dimension 'time' or 'feat'
        device str: devide

    '''

    def __new__(self, model, NumTimeSteps, NumFeatures, method='GRAD',mode='time', device='cpu'):
        if isinstance(model, torch.nn.Module):

            return Saliency_PTY(model, NumTimeSteps, NumFeatures, method=method,mode=mode, device=device)
        
        elif isinstance(model, tensorflow.keras.Model):

            return Saliency_TF(model, NumTimeSteps, NumFeatures, method=method,mode=mode)
        else: 
            raise NotImplementedError("Please use a TF or PYT Classification model! If the current model is a TF or PYT Model, try calling the wrappers directly (TF -> Saliency_TF, PYT -> Saliency_PYT)")
        
