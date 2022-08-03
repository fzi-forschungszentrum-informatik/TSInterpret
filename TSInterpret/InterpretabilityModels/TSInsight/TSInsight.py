from zmq import device
from TSInterpret.InterpretabilityModels.FeatureAttribution import FeatureAttribution
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

class TSInsight(FeatureAttribution):
    '''
        Instantiated TS Insight
        mlmodel: Machine Learning Model to be explained.
        mode : Second dimension is feature --> 'feat', is time --> 'time'
        backend: PYT or TF
        autoencode: None or instance of an implemented and traine AE 
        data: In case of TF a Tuple, in case of PYT a DataLoader 
    '''
    def __init__(self, mlmodel,shape, mode='feat', backend='PYT',autoencoder = None):
        super().__init__(mlmodel,mode)
        self.mode = mode
        self.shape =shape
        self.autoencoder = autoencoder
        self.backend=backend

    def explain(self, item, flatten=True):
        pass

    def plot(self,original,exp,all_in_one=True,vis_change=False,save=None ):
        print(exp.shape)
        print(original.shape)
        if all_in_one:
            i=0
            for a in original:

                ax011 = plt.subplot(1,1,1)
                #ax012 = ax011.twinx()
                #sal_02= np.abs(original[i].reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
                #if vis_change:
                #    sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
                #else: 
                #    sns.heatmap(np.zeros_like(sal_02),fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
                sns.lineplot(x=range(0,len(original.reshape(-1))), y=original.flatten(), color='red',legend=False, label='Original')
                sns.lineplot(x=range(0,len(original.reshape(-1))), y=exp.flatten(), color='blue', legend=False, label='Supressed')
                plt.legend()
                i=i+1
       
        else:
            ax011 = plt.subplot(2,1,1)
            ax012 = ax011.twinx()
            sal_02= np.abs(original.reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
            if vis_change:
                sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
            else: 
                sns.heatmap(np.zeros_like(sal_02),fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
            p=sns.lineplot(x=range(0,len(original.reshape(-1))), y=original.flatten(), color='white', ax=ax012)
            p.set_ylabel(f"Original")

            ax031 = plt.subplot(2,1,2)
            ax032 = ax031.twinx()
            sal_02= np.abs(original.reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
            if vis_change:
                sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax031, yticklabels=False)
            else: 
                sns.heatmap(np.zeros_like(sal_02),fmt="g", cmap='viridis', cbar=False, ax=ax031, yticklabels=False)
            p=sns.lineplot(x=range(0,len(original.reshape(-1))), y=exp.flatten(), color='white', ax=ax032)
            p.set_ylabel(f"Supressed")
        if save== None:
            plt.show()
        else: 
            plt.savefig(save)
