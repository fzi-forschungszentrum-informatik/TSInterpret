import matplotlib.pyplot as plt
import seaborn as sns

from TSInterpret.InterpretabilityModels.FeatureAttribution import FeatureAttribution


class Saliency(FeatureAttribution):
    """
    Base Method for Saliency Calculation based on [1].
    Please use the designated Subclasses SaliencyMethods_PYT.py
    for PyTorch explanations and SaliencyMethods_TF.py
    for Tensforflow explanations.

    References
    ----------
    [1] Ismail, Aya Abdelsalam, et al.
    "Benchmarking deep learning interpretability in time series predictions."
    Advances in neural information processing systems 33 (2020): 6441-6452.
    ----------
    """

    def __init__(
        self,
        model,
        NumTimeSteps: int,
        NumFeatures: int,
        method: str = "GRAD",
        mode: str = "time",
    ) -> None:
        """
        Arguments:
            model [torch.nn.Module,tf.keras.models]: model to be explained.
            NumTimeSteps int: number of timesteps.
            NumFeatures int: number of features.
            method str: Saliency Method to be used.
            mode str: Second dimension 'time'->`(1,time,feat)`  or 'feat'->`(1,feat,time)`.
        """
        super().__init__(model, mode)
        self.NumTimeSteps = NumTimeSteps
        self.NumFeatures = NumFeatures
        self.method = method

    def explain(self):
        raise NotImplementedError("Don't use the base CF class directly")

    def plot(self, item, exp, figsize=(15, 15), heatmap=False, save=None):
        """
        Plots explanation on the explained Sample.

        Arguments:
            item np.array: instance to be explained,if `mode = time`->`(1,time,feat)`  or `mode = feat`->`(1,feat,time)`.
            exp np.array: explanation, ,if `mode = time`->`(time,feat)`  or `mode = feat`->`(feat,time)`.
            figsize (int,int): desired size of plot.
            heatmap bool: 'True' if only heatmap, otherwise 'False'.
            save str: Path to save figure.
        """
        plt.style.use("classic")
        i = 0
        if self.mode == "time":
            print("time mode")
            item = item.reshape(1, item.shape[2], item.shape[1])
            exp = exp.reshape(exp.shape[-1], -1)
        else:
            print("NOT Time mode")

        if heatmap:
            ax011 = plt.subplot(1, 1, 1)
            sns.heatmap(
                exp,
                fmt="g",
                cmap="viridis",
                cbar=True,
                ax=ax011,
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
        elif len(item[0]) == 1:
            # if only onedimensional input
            fig, axn = plt.subplots(len(item[0]), 1, sharex=True, sharey=True)
            # cbar_ax = fig.add_axes([.91, .3, .03, .4])
            axn012 = axn.twinx()
            sns.heatmap(
                exp.reshape(1, -1),
                fmt="g",
                cmap="viridis",
                ax=axn,
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
            sns.lineplot(
                x=range(0, len(item[0][0].reshape(-1))),
                y=item[0][0].flatten(),
                ax=axn012,
                color="white",
            )
        else:
            ax011 = []

            fig, axn = plt.subplots(len(item[0]), 1, sharex=True, sharey=True)
            cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

            for channel in item[0]:
                # print(item.shape)
                # ax011.append(plt.subplot(len(item[0]),1,i+1))
                # ax012.append(ax011[i].twinx())
                # ax011[i].set_facecolor("#440154FF")
                axn012 = axn[i].twinx()

                sns.heatmap(
                    exp[i].reshape(1, -1),
                    fmt="g",
                    cmap="viridis",
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    ax=axn[i],
                    yticklabels=False,
                    vmin=0,
                    vmax=1,
                )
                sns.lineplot(
                    x=range(0, len(channel.reshape(-1))),
                    y=channel.flatten(),
                    ax=axn012,
                    color="white",
                )
                plt.xlabel("Time", fontweight="bold", fontsize="large")
                plt.ylabel(f"Feature {i}", fontweight="bold", fontsize="large")
                i = i + 1
            fig.tight_layout(rect=[0, 0, 0.9, 1])
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
