import pandas as pd
import numpy as np
import torch
import gc
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random
from sklearn import tree
from sklearn.model_selection import cross_val_score
#import UCRDataset

class UCRDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label

import torch
from torch import nn
from typing import cast, Any, Dict, List, Tuple, Optional
import numpy as np
class ResNetBaseline(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


import torch
from torch import nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)

def fit(model, train_loader, val_loader, num_epochs: int = 1500,
            val_size: float = 0.2, learning_rate: float = 0.001,
            patience: int = 100) -> None: # patience war 10 
        """Trains the inception model
        Arguments
        ----------
        batch_size:
            Batch size to use for training and validation
        num_epochs:
            Maximum number of epochs to train for
        val_size:
            Fraction of training set to use for validation
        learning_rate:
            Learning rate to use with Adam optimizer
        patience:
            Maximum number of epochs to wait without improvement before
            early stopping
        """
        #train_loader, val_loader = self.get_loaders(batch_size, mode='train', val_size=val_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_val_loss = np.inf
        patience_counter = 0
        best_state_dict = None
        train_loss_array=[]
        val_loss_array=[]
        model.train()
        for epoch in range(num_epochs):
            epoch_train_loss = []
            for x_t, y_t in train_loader:
                optimizer.zero_grad()
                output = model(x_t.float())
                if len(y_t.shape) == 1:
                    train_loss = F.binary_cross_entropy_with_logits(
                        output, y_t.unsqueeze(-1).float(), reduction='mean'
                    )
                else:
                    train_loss = F.cross_entropy(output, y_t.argmax(dim=-1), reduction='mean')

                epoch_train_loss.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
            train_loss_array.append(np.mean(epoch_train_loss))

            epoch_val_loss = []
            model.eval()
            for x_v, y_v in  val_loader:
                with torch.no_grad():
                    output = model(x_v.float())
                    if len(y_v.shape) == 1:
                        val_loss = F.binary_cross_entropy_with_logits(
                            output, y_v.unsqueeze(-1).float(), reduction='mean'
                        ).item()
                    else:
                        val_loss = F.cross_entropy(output,
                                                   y_v.argmax(dim=-1), reduction='mean').item()
                    epoch_val_loss.append(val_loss)
            val_loss_array.append(np.mean(epoch_val_loss))

            print(f'Epoch: {epoch + 1}, '
                  f'Train loss: {round(train_loss_array[-1], 3)}, '
                  f'Val loss: {round(val_loss_array[-1], 3)}')

            if val_loss_array[-1] < best_val_loss:
                best_val_loss = val_loss_array[-1]
                best_state_dict = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter == patience:
                    if best_state_dict is not None:
                        model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                    print('Early stopping!')
                    return None

def get_all_preds(model, loader):
    model.eval()
    with torch.no_grad():
        all_preds = []
        labels = []
        for batch in loader:
            item, label = batch
            preds = model(item.float())
            all_preds = all_preds + preds.argmax(dim=1).tolist()
            labels = labels + label.tolist()
    return all_preds, labels



def objective(trial, train_dataset, test_dataset,output=5):
    kernel_size = trial.suggest_int('kernel_size', 5, 25, 5)
    stride = 1
    padding = kernel_size - 1
    batch_size = trial.suggest_int('batch_size', 4, 32, 4)
    epochs = trial.suggest_int('n_epochs', 300, 1500, 100)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True) 
    x,y= train_dataset[0]
    input_size=x.shape[-1]
    device = torch.device( "cpu")#"cuda:0" if torch.cuda.is_available() else
    model = CNN_TSNet(kernel_size=kernel_size, stride=stride, padding=padding,input_size= input_size, output=output,out_channels=output).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    #train_dataset = UCRDataset(train_x,train_y)
    #test_dataset = UCRDataset(test_x,test_y)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for idx, (inputs,labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
        
            loss = criterion(preds,labels)
            loss.backward()
            optimizer.step()
        gc.collect()
        
    test_preds, ground_truth = get_all_preds(model, test_loader)
    return accuracy_score(ground_truth, test_preds)

if __name__=='__main__':
    '''Parameters'''
    stride = 1
    kernel_size=10
    padding = kernel_size - 1
    batch_size = 4  # 5
    lr = 0.00026          # 1e-6
    epochs = 300    # 1500

    '''Training Data '''
    train = pd.read_csv('/media/jacqueline/Data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv', sep='\t', header=None)
    test = pd.read_csv('/media/jacqueline/Data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv', sep='\t', header=None)
    train_y, train_x = train.loc[:, 0].apply(lambda x: x-1).to_numpy(), train.loc[:, 1:].to_numpy()
    test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1])) 
    train_dataset = UCRDataset(train_x,train_y)
    test_dataset = UCRDataset(test_x,test_y)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

    '''Model'''

    device = torch.device("cpu")#"cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_TSNet(kernel_size=kernel_size, stride=stride, padding=padding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    '''Parameter Objective'''
    study = optuna.create_study(direction='maximize')
    study.optimize(model.objective, n_trials=300)
    train_losses = []
    valid_losses = []
    for epoch in tqdm(range(epochs)):
        tl = model.train()
        train_losses.append(tl.detach().numpy())
        vl = model.valid()
        valid_losses.append(vl.detach().numpy())
        gc.collect()

    '''Run Plots'''
    sns.set(rc={'figure.figsize':(5,4)})
    sns.lineplot(y=[x.item() for x in train_losses], x=range(len(train_losses)), label='train')
    sns.lineplot(y=[x.item() for x in valid_losses], x=range(len(valid_losses)), label='test')
    #TODO Save Function
    test_preds, ground_truth = model.get_all_preds(model, test_loader)

    sns.set(rc={'figure.figsize':(5,4)})
    heatmap=confusion_matrix(ground_truth, test_preds)
    sns.heatmap(heatmap, annot=True)
    #TODO Save Function
    accuracy_score(ground_truth, test_preds)
    #TODO Save Function 
    torch.save(model.state_dict(), 'gunpoint_best_state_test')