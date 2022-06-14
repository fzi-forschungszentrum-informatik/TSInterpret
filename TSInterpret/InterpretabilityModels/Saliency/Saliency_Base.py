
from TSInterpret.InterpretabilityModels.FeatureAttribution import FeatureAttribution


class Saliency(FeatureAttribution):
    '''
    Base Method for Saliency Calculation based on [1]. Please use the designated Subclasses SaliencyMethods_PYT.py for PyTorch explanations and SaliencyMethods_TF.py for Tensforflow explanations.

    [1] Ismail, Aya Abdelsalam, et al. "Benchmarking deep learning interpretability in time series predictions." Advances in neural information processing systems 33 (2020): 6441-6452.
    '''
    def __init__(self, model, NumTimeSteps, NumFeatures, method='GRAD',mode='time',backend= 'torch',device='cpu') -> None:
        super().__init__(model, mode)
        self.NumTimeSteps=NumTimeSteps
        self.NumFeatures=NumFeatures 
        self.method = method
        print('Mode in Saliency', self.mode)

    def explain(self):
        pass

    def plot_heatmap(self):
        #TODO Move HeatMap to here
        pass 

