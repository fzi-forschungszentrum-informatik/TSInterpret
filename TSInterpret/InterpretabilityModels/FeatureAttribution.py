import numpy as np
from TSInterpret.InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
import matplotlib.pyplot as plt
import seaborn as sns 
from abc import ABC, abstractmethod
class FeatureAttribution(InterpretabilityBase):
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

    def __init__(self, mlmodel,mode):
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

    def plot(self, item, exp, figsize=(15,15),heatmap= False, normelize_saliency = True,  save = None):
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
            Visualization of Explanation
        """
        plt.style.use("classic")
        colors = [
            '#08F7FE',  # teal/cyan
            '#FE53BB',  # pink
            '#F5D300',  # yellow
            '#00ff41',  # matrix green
            ]
            #Figure out number changed channels
            #index= np.where(np.any(item))
        i=0
        if self.mode=='time':
            print('time mode')
            item =item.reshape(1,item.shape[2],item.shape[1])
            print(exp.shape)
            #TODO Excluded because of LEFTIST
           # exp =exp.reshape(exp.shape[-1],-1)
            print(exp.shape)
        else: 
            print(self.mode)
            print('NOT Time mode')
        vmin=np.min(exp)
        vmax=np.max(exp)
        if heatmap: 
            ax011 = plt.subplot(1,1,1)
            ax012 = ax011.twinx()
            if normelize_saliency:
                sns.heatmap(exp, fmt="g", cmap='viridis', cbar=True, ax=ax011, yticklabels=False, vmin=vmin,vmax=vmax)
            else: 
                sns.heatmap(exp, fmt="g", cmap='viridis', cbar=True, ax=ax011, yticklabels=False)
        else: 
            ax011=[]
            ax012=[]
            for channel in item[0]:
                #print(item.shape)
                ax011.append(plt.subplot(len(item[0]),1,i+1))
                ax012.append(ax011[i].twinx())
                print(i)
                if normelize_saliency:
                    sns.heatmap(exp[i].reshape(1,-1), fmt="g", cmap='viridis', cbar=False, ax=ax011[i], yticklabels=False, vmin=vmin,vmax=vmax)
                else: 
                    sns.heatmap(exp[i].reshape(1,-1), fmt="g", cmap='viridis', cbar=False, ax=ax011[i], yticklabels=False)
                sns.lineplot(x=range(0,len(channel.reshape(-1))), y=channel.flatten(),ax=ax012[i])
                plt.xlabel('Time', fontweight = 'bold', fontsize='large')
                plt.ylabel('Value', fontweight = 'bold', fontsize='large')
                i=i+1

        if save == None: 
            plt.show()
        else: 
            plt.savefig(save)
