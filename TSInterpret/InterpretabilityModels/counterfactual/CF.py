#TODO From WildBoar
#Karlsson, I., Rebane, J., Papapetrou, P. et al. Locally and globally explainable time series tweaking. Knowl Inf Syst 62, 1671–1700 (2020)
#Only works with Shaplet Forest classifier, KNearestNeighbour Classifier 

from InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod
import numpy as np 
import pandas as pd
from typing import Tuple

class CF(InterpretabilityBase):
    """
    Abstract class to implement Coutnterfactual Methods for time series data 
    model: Machine Learning Model to be explained.
    mode : Second dimension is feature --> 'feat', is time --> 'time'
    """

    def __init__(self, mlmodel,mode) -> None:
        super().__init__(mlmodel)
        #self.model_to_explain = mlmodel
        self.mode=mode

    def explain(self) -> Tuple[np.array, int]:
        """
        Explains instance or model. 
        Parameters
        ----------
        instance: np.array
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).
        Returns
        -------
        Tuple(cf, label)
        """
        raise NotImplementedError("Please don't use the base class directly")

    def plot (self, original,org_label, exp,exp_label, vis_change= True,all_in_one=False, save=None):
        """
        Basic Plot Function for visualizing Coutnerfactuals.
        Parameters
        ----------
        instance: np.array
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).
        Returns
        -------
        pd.DataFrame
            Encoded and normalised counterfactual examples.
        """
        
        if all_in_one:
            ax011 = plt.subplot(1,1,1)
            ax012 = ax011.twinx()
            sal_02= np.abs(original.reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
            if vis_change:
                sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
            else: 
                sns.heatmap(np.zeros_like(sal_02),fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
            sns.lineplot(x=range(0,len(original.reshape(-1))), y=original.flatten(), color='white', ax=ax012,legend=False, label='Original')
            sns.lineplot(x=range(0,len(original.reshape(-1))), y=exp.flatten(), color='black', ax=ax012,legend=False, label='Counterfactual')
            plt.legend()
       
        else:
            ax011 = plt.subplot(2,1,1)
            ax012 = ax011.twinx()
            sal_02= np.abs(original.reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
            if vis_change:
                sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
            else: 
                sns.heatmap(np.zeros_like(sal_02),fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)

            p=sns.lineplot(x=range(0,len(original.reshape(-1))), y=original.flatten(), color='white', ax=ax012,label=f"{org_label}")
            p.set_ylabel(f"Original")
           
            ax031 = plt.subplot(2,1,2)
            ax032 = ax031.twinx()
            sal_02= np.abs(original.reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
            if vis_change:
                sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax031, yticklabels=False)
            else: 
                sns.heatmap(np.zeros_like(sal_02),fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)

            p=sns.lineplot(x=range(0,len(original.reshape(-1))), y=exp.flatten(), color='white', ax=ax032,label=f"{exp_label}")
            p.set_ylabel(f"Counterfactual")
        if save == None: 
            plt.show()
        else:
            plt.savefig(save)
    
    def plot_in_one(self,item,org_label,exp,cf_label):
        """
        Basic Plot Function for Visualizing Coutnerfactuals.
        Parameters
        ----------
        instance: np.array
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).
        Returns
        -------
        pd.DataFrame
            Encoded and normalised counterfactual examples.
        """
        
        plt.style.use("classic")
        colors = [
            '#08F7FE',  # teal/cyan
            '#FE53BB',  # pink
            '#F5D300',  # yellow
            '#00ff41',  # matrix green
        ]
        indices= np.where(exp[0] != item)
        df = pd.DataFrame({f'Predicted: {org_label}': list(item.flatten()),
                   f'Counterfactual: {cf_label}': list(exp.flatten())})
        fig, ax = plt.subplots(figsize=(10,5))
        df.plot(marker='.', color=colors, ax=ax)
        # Redraw the data with low alpha and slighty increased linewidth:
        n_shades = 10
        diff_linewidth = 1.05
        alpha_value = 0.3 / n_shades
        for n in range(1, n_shades+1):
            df.plot(marker='.',
            linewidth=2+(diff_linewidth*n),
            alpha=alpha_value,
            legend=False,
            ax=ax,
            color=colors)

        ax.grid(color='#2A3459')
        plt.xlabel('Time', fontweight = 'bold', fontsize='large')
        plt.ylabel('Value', fontweight = 'bold', fontsize='large')
        #plt.savefig('../Images/Initial_Example_Neon.pdf')
        plt.show()