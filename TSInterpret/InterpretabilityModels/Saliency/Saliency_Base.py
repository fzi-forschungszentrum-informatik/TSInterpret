
from TSInterpret.InterpretabilityModels.FeatureAttribution import FeatureAttribution


class Saliency(FeatureAttribution):
    '''
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

