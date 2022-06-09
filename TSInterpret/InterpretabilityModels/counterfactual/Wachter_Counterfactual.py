from InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import seaborn as sns 
import matplotlib.pyplot as plt

DECISION_THRESHOLD = 0.5

class W_CF(InterpretabilityBase):
    """

    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    Code: Adapted to time-series from tf_explain 
    TODO 
    -------
        * Only Tensorflow 
        * Is the Implementation Suitable ? 
        * Works for all Models or only FCN ? 
    Restrictions: 
     - Model needs to have a Gradient ! 
    -------
    """

    def __init__(self, mlmodel,mode):
        self.model_to_explain = mlmodel
        self.mode = mode 

  

    def explain(  self, x: np.ndarray,  y_target: List[int],  lr: float = 0.01,  lambda_param: float = 0.01,    n_iter: int = 1000,   t_max_min: float = 1.0,
    norm: int = 1,    clamp: bool = True,    loss_type: str = "MSE",) -> np.ndarray:
        """
        Generates counterfactual example according to Wachter et.al for input instance x
        Parameters
        ----------
        x: factual to explain
        lr: learning rate for gradient descent
        lambda_param: weight factor for feature_cost
        y_target: List of one-hot-encoded target class
        n_iter: maximum number of iteration
        t_max_min: maximum time of search
        norm: L-norm to calculate cost
        clamp: If true, feature values will be clamped to (0, 1)
        loss_type: String for loss function (MSE or BCE)
        Returns
        -------
        Counterfactual example as np.ndarray
        """
        device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
        # returns counterfactual instance
        torch.manual_seed(0)

        #if feature_costs is not None:
        #    feature_costs = torch.from_numpy(feature_costs).float().to(device)

        x = torch.from_numpy(x).float().to(device)
        #y_target = torch.tensor(y_target).float().to(device)
        lamb = torch.tensor(lambda_param).float().to(device)
        # x_new is used for gradient search in optimizing process
        x_new = Variable(x.clone(), requires_grad=True)
        # x_new_enc is a copy of x_new with reconstructed encoding constraints of x_new
        # such that categorical data is either 0 or 1
        #print(x_new.shape)
        x_new_enc =x_new# reconstruct_encoding_constraints(
        #    x_new, cat_feature_indices, binary_cat_features
        #)

        optimizer = optim.Adam([x_new], lr, amsgrad=True)
        softmax = nn.Softmax()

        if loss_type == "MSE":
            loss_fn = torch.nn.MSELoss()
        
        #torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        #print(softmax(torch_model(x_new)))
            f_x_new=self.model_to_explain(x_new)
            #f_x_new = softmax(torch_model(x_new))#.detach().numpy()#[0][y_target]
        #One Hot Encode y_target
        #print(f_x_new.shape[1])
        #print(torch.tensor([y_target]))
            y_target_proba=torch.nn.functional.one_hot(torch.tensor([y_target]),num_classes=f_x_new.shape[1]).float()

        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=t_max_min)

        while f_x_new[0][y_target] <= DECISION_THRESHOLD:
            it = 0
            while f_x_new[0][y_target]<= 0.5 and it < n_iter:
                optimizer.zero_grad()
                x_new_enc = x_new 
                f_x_new=self.model_to_explain(x_new)
                #f_x_new = softmax(torch_model(x_new_enc))#[:, 1]
                cost = (
                    torch.dist(x_new_enc, x, norm)
                )
                loss = loss_fn(f_x_new, y_target_proba) + lamb * cost
                loss.backward()
                optimizer.step()
                it += 1
            lamb -= 0.05

            if datetime.datetime.now() - t0 > t_max:
                print("Timeout - No Counterfactual Explanation Found")
                return None
            elif f_x_new[0][y_target] >= 0.5:
                print("Counterfactual Explanation Found")
        return x_new_enc.cpu().detach().numpy().squeeze(axis=0)
    
    def plot (self, original, exp, vis_change= True,all_in_one=False, save=None):
        
        if all_in_one:
            ax011 = plt.subplot(1,1,1)
            ax012 = ax011.twinx()
            sal_02= np.abs(original.reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
            if vis_change:
                sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
            sns.lineplot(x=range(0,len(original.reshape(-1))), y=original.flatten(), color='white', ax=ax012,legend=False, label='Original')
            sns.lineplot(x=range(0,len(original.reshape(-1))), y=exp.flatten(), color='black', ax=ax012,legend=False, label='Counterfactual')
            plt.legend()
        
        else:
            ax011 = plt.subplot(2,1,1)
            ax012 = ax011.twinx()
            sal_02= np.abs(original.reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
            if vis_change:
                sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
                p=sns.lineplot(x=range(0,len(original.reshape(-1))), y=original.flatten(), color='white', ax=ax012,legend=False)
                p.set_ylabel("Original")

            ax031 = plt.subplot(2,1,2)
            ax032 = ax031.twinx()
            sal_02= np.abs(original.reshape(-1)-np.array(exp).reshape(-1)).reshape(1,-1)
            if vis_change:
                sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax031, yticklabels=False)
            p=sns.lineplot(x=range(0,len(original.reshape(-1))), y=exp.flatten(), color='white', ax=ax032,legend=False)
            p.set_ylabel("Counterfactual")
        if save == None: 
            plt.show()
        else:
            plt.savefig(save)

