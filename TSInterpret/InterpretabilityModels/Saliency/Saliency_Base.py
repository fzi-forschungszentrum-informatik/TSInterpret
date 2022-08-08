
from TSInterpret.InterpretabilityModels.FeatureAttribution import FeatureAttribution
import seaborn as sns
import matplotlib.pyplot as plt

class Saliency(FeatureAttribution):
    '''
    Base Method for Saliency Calculation based on [1]. Please use the designated Subclasses SaliencyMethods_PYT.py for PyTorch explanations 
    and SaliencyMethods_TF.py for Tensforflow explanations.

    [1] Ismail, Aya Abdelsalam, et al. "Benchmarking deep learning interpretability in time series predictions." Advances in neural information processing systems 33 (2020): 6441-6452.
    '''
    def __init__(self, model, NumTimeSteps, NumFeatures, method='GRAD',mode='time') -> None:
        '''
        Arguments:
            model: model to be explained. 
            NumTimeSteps int: number of timesteps. 
            NumFeature int: number of features.
            method str: Saliency Method to be used.
            mode str: second dimension is 'feat' or 'time'.
        '''
        super().__init__(model, mode)
        self.NumTimeSteps=NumTimeSteps
        self.NumFeatures=NumFeatures 
        self.method = method

    def explain(self):
        raise NotImplementedError("Please don't use the base CF class directly")
    
    def plot(self, item, exp, figsize=(15,15),heatmap= False, save = None):
        """
        Plots expalantion on the explained Sample. 

        Arguments: 
            item np.array: instance to be explained.
            exp np.array: expalantaion.
            figsize (int,int): desired size of plot. 
            heatmap bool: 'True' if only heatmap, otherwise 'False'.
            save str: Path to save figure. 
        """ 
        plt.style.use("classic")
        colors = [
            '#08F7FE',  # teal/cyan
            '#FE53BB',  # pink
            '#F5D300',  # yellow
            '#00ff41',  # matrix green
            ]
        cmap =sns.diverging_palette(220,251,s=74,l=50, n=16)
        i=0
        if self.mode=='time':
            print('time mode')
            item =item.reshape(1,item.shape[2],item.shape[1])
            exp =exp.reshape(exp.shape[-1],-1)
        else: 
            print('NOT Time mode')

        if heatmap:
            pass 
            ax011 = plt.subplot(1,1,1)
            ax012 = ax011.twinx()
            sns.heatmap(exp, fmt="g", cmap='viridis', cbar=True, ax=ax011, yticklabels=False,vmin=0,vmax=1)
        else: 
            ax011=[]
            ax012=[]

            fig, axn = plt.subplots(len(item[0]), 1, sharex=True, sharey=True)
            cbar_ax = fig.add_axes([.91, .3, .03, .4])

            for channel in item[0]:
                #print(item.shape)
                #ax011.append(plt.subplot(len(item[0]),1,i+1))
                #ax012.append(ax011[i].twinx())
                #ax011[i].set_facecolor("#440154FF")
                axn012=axn[i].twinx()
                

                sns.heatmap(exp[i].reshape(1,-1), fmt="g",cmap='viridis', cbar=i == 0 ,cbar_ax=None if i else cbar_ax, ax=axn[i], yticklabels=False, vmin=0,vmax=1)
                sns.lineplot(x=range(0,len(channel.reshape(-1))), y=channel.flatten(),ax=axn012,color='white')
                plt.xlabel('Time', fontweight = 'bold', fontsize='large')
                plt.ylabel(f'Feature {i}', fontweight = 'bold', fontsize='large')
                i=i+1
            fig.tight_layout(rect=[0, 0, .9, 1])
        if save == None: 
            plt.show()
        else: 
            plt.savefig(save)

