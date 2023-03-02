from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from TSInterpret.InterpretabilityModels.InstanceBase import InstanceBase


class CF(InstanceBase):
    """
    Abstract class to implement Coutnterfactual Methods for time series data
    """

    def __init__(self, mlmodel, mode) -> None:
        """Initialization of CF.
        Arguments:
            mlmodel [torch.nn.Module, Callabale, tf.keras.Model]: Machine Learning Model to be explained.
            mode str : Second dimension is feature --> 'feat', is time --> 'time'
        """

        super().__init__(mlmodel, mode)

    def explain(self) -> Tuple[np.array, int]:
        """
        Explains instance or model.
        """
        raise NotImplementedError("Please don't use the base CF class directly")

    def plot(
        self,
        original,
        org_label,
        exp,
        exp_label,
        vis_change=True,
        all_in_one=False,
        save_fig=None,
    ):
        """
        Basic Plot Function for visualizing Coutnerfactuals.
        Arguments:
            original np.array: Instance to be explained. Shape: `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`
            org_label int: Label of instance to be explained.
            exp np.array: Explanation. `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`
            exp_label int: Label of Explanation.
            vis_change bool: Change to be visualized as heatmap.
            all_in_one bool: Original and Counterfactual in one plot.
            save_fig str: Path to save fig at.
        """

        if all_in_one:
            ax011 = plt.subplot(1, 1, 1)
            ax012 = ax011.twinx()
            sal_02 = np.abs(original.reshape(-1) - np.array(exp).reshape(-1)).reshape(
                1, -1
            )
            if vis_change:
                sns.heatmap(
                    sal_02,
                    fmt="g",
                    cmap="viridis",
                    cbar=False,
                    ax=ax011,
                    yticklabels=False,
                )
            else:
                sns.heatmap(
                    np.zeros_like(sal_02),
                    fmt="g",
                    cmap="viridis",
                    cbar=False,
                    ax=ax011,
                    yticklabels=False,
                )
            sns.lineplot(
                x=range(0, len(original.reshape(-1))),
                y=original.flatten(),
                color="white",
                ax=ax012,
                legend=False,
                label="Original",
            )
            sns.lineplot(
                x=range(0, len(original.reshape(-1))),
                y=exp.flatten(),
                color="black",
                ax=ax012,
                legend=False,
                label="Counterfactual",
            )
            plt.legend()

        else:
            ax011 = plt.subplot(2, 1, 1)
            ax012 = ax011.twinx()
            sal_02 = np.abs(original.reshape(-1) - np.array(exp).reshape(-1)).reshape(
                1, -1
            )
            if vis_change:
                sns.heatmap(
                    sal_02,
                    fmt="g",
                    cmap="viridis",
                    cbar=False,
                    ax=ax011,
                    yticklabels=False,
                )
            else:
                sns.heatmap(
                    np.zeros_like(sal_02),
                    fmt="g",
                    cmap="viridis",
                    cbar=False,
                    ax=ax011,
                    yticklabels=False,
                )

            p = sns.lineplot(
                x=range(0, len(original.reshape(-1))),
                y=original.flatten(),
                color="white",
                ax=ax012,
                label=f"{org_label}",
            )
            p.set_ylabel("Original")

            ax031 = plt.subplot(2, 1, 2)
            ax032 = ax031.twinx()
            sal_02 = np.abs(original.reshape(-1) - np.array(exp).reshape(-1)).reshape(
                1, -1
            )
            if vis_change:
                sns.heatmap(
                    sal_02,
                    fmt="g",
                    cmap="viridis",
                    cbar=False,
                    ax=ax031,
                    yticklabels=False,
                )
            else:
                sns.heatmap(
                    np.zeros_like(sal_02),
                    fmt="g",
                    cmap="viridis",
                    cbar=False,
                    ax=ax011,
                    yticklabels=False,
                )

            p = sns.lineplot(
                x=range(0, len(original.reshape(-1))),
                y=exp.flatten(),
                color="white",
                ax=ax032,
                label=f"{exp_label}",
            )
            p.set_ylabel("Counterfactual")
        if save_fig is None:
            plt.show()
        else:
            plt.savefig(save_fig)

    def plot_in_one(
        self, item, org_label, exp, cf_label, save_fig=None, figsize=(15, 15)
    ):
        """
        Plot Function for Counterfactuals in uni-and multivariate setting. In the multivariate setting only the changed features are visualized.
        Arguments:
            item np.array: original instance. Shape: `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`
            org_label int: originally predicted label.
            exp np.array: returned explanation. Shape: `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`
            cf_label int: lebel of returned instance.
            figsize Tuple: Size of Figure (x,y).
            save_fig str: Path to Save the figure.
        """
        if self.mode == "time":
            item = item.reshape(item.shape[0], item.shape[2], item.shape[1])
        # TODO This is new and needs to be testes
        ind = ""
        # print("Item Shape", item.shape[-2])
        if item.shape[-2] > 1:

            res = (item != exp).any(-1)
            # print(res)
            ind = np.where(res)
            # print(ind)
            if len(ind[0]) == 0:
                print("Items are identical")
                return
            elif len(ind[0]) > 1:
                self.plot_multi(
                    item, org_label, exp, cf_label, figsize=figsize, save_fig=save_fig
                )
                return
            else:
                item = item[ind]

        plt.style.use("classic")
        colors = [
            "#08F7FE",  # teal/cyan
            "#FE53BB",  # pink
            "#F5D300",  # yellow
            "#00ff41",  # matrix green
        ]
        df = pd.DataFrame(
            {
                f"Predicted: {org_label}": list(item.flatten()),
                f"Counterfactual: {cf_label}": list(exp.flatten()),
            }
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        df.plot(marker=".", color=colors, ax=ax)
        # Redraw the data with low alpha and slighty increased linewidth:
        n_shades = 10
        diff_linewidth = 1.05
        alpha_value = 0.3 / n_shades
        for n in range(1, n_shades + 1):
            df.plot(
                marker=".",
                linewidth=2 + (diff_linewidth * n),
                alpha=alpha_value,
                legend=False,
                ax=ax,
                color=colors,
            )

        ax.grid(color="#2A3459")
        plt.xlabel("Time", fontweight="bold", fontsize="large")
        if ind != "":
            plt.ylabel(f"Feature {ind[0][0]}", fontweight="bold", fontsize="large")
        else:
            plt.ylabel("Value", fontweight="bold", fontsize="large")
        if save_fig is None:
            plt.show()
        else:
            plt.savefig(save_fig)

    def plot_multi(
        self, item, org_label, exp, cf_label, figsize=(15, 15), save_fig=None
    ):
        """Plot Function for Ates et al., used if multiple features are changed in a Multivariate Setting.
        Also called via plot_in_one. Preferably, do not use directly.
        Arguments:
            item np.array: original instance. Shape: `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`
            org_label int: originally predicted label.
            exp np.array: returned explanation. Shape: `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`
            cf_label int: lebel of returned instance.
            figsize Tuple: Size of Figure (x,y).
            save_fig str: Path to Save the figure.
        """
        # if self.mode == 'time':
        #    item = item.reshape(item.shape[0],item.shape[2],item.shape[1])

        plt.style.use("classic")
        colors = [
            "#08F7FE",  # teal/cyan
            "#FE53BB",  # pink
            "#F5D300",  # yellow
            "#00ff41",  # matrix green
        ]
        # Figure out number changed channels
        # index= np.where(np.any(item))

        i = 0
        res = (item != exp).any(-1)
        ind = np.where(res)
        if len(ind[0]) == 0:
            print("Items are identical")
            return

        # Draw changed channels
        fig, ax = plt.subplots(len(ind[0]), 1, figsize=figsize)

        # print(ax)
        for channel in ind[0]:
            # fig,ax=plt.subplot(len(ind[0]),1,i)

            df = pd.DataFrame(
                {
                    f"Predicted: {org_label}": list(item[channel].flatten()),
                    f"Counterfactual: {cf_label}": list(exp[channel].flatten()),
                }
            )

            df.plot(marker=".", color=colors, ax=ax[i])
            # Redraw the data with low alpha and slighty increased linewidth:
            n_shades = 10
            diff_linewidth = 1.05
            alpha_value = 0.3 / n_shades
            for n in range(1, n_shades + 1):
                df.plot(
                    marker=".",
                    linewidth=2 + (diff_linewidth * n),
                    alpha=alpha_value,
                    legend=False,
                    ax=ax[i],
                    color=colors,
                )
            ax[i].grid(color="#2A3459")
            plt.xlabel("Time", fontweight="bold", fontsize="large")
            ax[i].set_ylabel(f"Feature {channel}", fontweight="bold", fontsize="large")
            i = i + 1
        if save_fig is None:
            plt.show()
        else:
            plt.savefig(save_fig)
