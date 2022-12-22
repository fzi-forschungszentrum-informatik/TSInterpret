from abc import abstractmethod
from locale import normalize

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing

from TSInterpret.InterpretabilityModels.InterpretabilityBase import InterpretabilityBase


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

    def __init__(self, mlmodel, mode):
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

    def plot(
        self,
        item,
        exp,
        figsize=(15, 15),
        heatmap=False,
        normelize_saliency=True,
        vmin=-1,
        vmax=1,
        save=None,
    ):
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
        #TODO CHeck with all typer
        """
        # TODO normelize on -1 to 1 and color appropriatly
        plt.style.use("classic")
        # Figure out number changed channels
        # index= np.where(np.any(item))
        # TODO Positiv8
        # cmap = pl.cm.RdBu

        # Get the colormap colors
        # my_cmap = cmap(np.arange(cmap.N))
        # Define the alphas in the range from 0 to 1
        # alphas = np.linspace(0, 1, cmap.N)
        # Define the background as white
        # BG = np.asarray([1., 1., 1.,])
        # Mix the colors with the background
        # for i in range(cmap.N):
        #    my_cmap[i,:-1] = my_cmap[i,:-1] * alphas[i] + BG * (1.-alphas[i])
        # Create new colormap which mimics the alpha values
        # my_cmap = ListedColormap(my_cmap)
        cmap = sns.diverging_palette(
            220, 251, s=74, l=50, n=16
        )  # 240, 10, center="dark")#sns.diverging_palette(250, 15, s=75, l=40,

        #                   n=9, center="dark")#sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
        i = 0
        if self.mode == "time":
            print("time mode")
            item = item.reshape(1, item.shape[2], item.shape[1])
            print(exp.shape)
            # TODO Excluded because of LEFTIST
            # exp =exp.reshape(exp.shape[-1],-1)
            print(exp.shape)
        else:
            print(self.mode)
            print("NOT Time mode")

        print(vmin)
        print(vmax)
        if normalize:
            # TODO Positive und negative werte einzeln Gewicxhten
            # TODO Customize color palet
            zeros = np.argwhere(exp.reshape(-1) == 0)
            shape = exp.shape
            exp = preprocessing.minmax_scale(exp.reshape(-1), feature_range=(-1, 1))
            np.put(exp, zeros, np.zeros_like(zeros))
            exp = exp.reshape(shape[0], shape[1])
        else:
            vmin = np.min(exp)
            vmax = np.max(exp)
        center = None
        if vmin < 0:
            center = 0
        if heatmap:
            pass
            ax011 = plt.subplot(1, 1, 1)
            ax012 = ax011.twinx()
            if normelize_saliency:
                sns.heatmap(
                    exp,
                    fmt="g",
                    cmap="viridis",
                    cbar=True,
                    ax=ax011,
                    yticklabels=False,
                    vmin=vmin,
                    vmax=vmax,
                    center=center,
                )
            else:
                sns.heatmap(
                    exp,
                    fmt="g",
                    cmap="viridis",
                    cbar=True,
                    ax=ax011,
                    yticklabels=False,
                    center=center,
                )
        else:
            ax011 = []
            ax012 = []

            for channel in item[0]:
                # print(item.shape)
                ax011.append(plt.subplot(len(item[0]), 1, i + 1))
                ax012.append(ax011[i].twinx())
                ax011[i].set_facecolor("#440154FF")

                print(i)
                if normelize_saliency:
                    # Used to be vivaris
                    sns.heatmap(
                        exp[i].reshape(1, -1),
                        fmt="g",
                        cmap=cmap,
                        cbar=True,
                        ax=ax011[i],
                        yticklabels=False,
                        vmin=vmin,
                        vmax=vmax,
                        center=center,
                    )
                else:
                    sns.heatmap(
                        exp[i].reshape(1, -1),
                        fmt="g",
                        cmap=cmap,
                        cbar=True,
                        ax=ax011[i],
                        yticklabels=False,
                        center=center,
                    )
                sns.lineplot(
                    x=range(0, len(channel.reshape(-1))),
                    y=channel.flatten(),
                    ax=ax012[i],
                )
                plt.xlabel("Time", fontweight="bold", fontsize="large")
                plt.ylabel("Value", fontweight="bold", fontsize="large")
                i = i + 1
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
