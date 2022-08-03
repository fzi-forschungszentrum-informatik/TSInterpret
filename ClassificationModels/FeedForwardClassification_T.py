import torch
from torch import nn
import torch.nn.functional as F
from typing import cast, Any, Dict, List, Tuple, Optional
import numpy as np
from ClassificationModels.CNN_T import ResNetBaseline, get_all_preds, fit, UCRDataset
class LinearBaseline(nn.Module):
    """A PyTorch implementation of the Linear Baseline
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_inputs: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_inputs': num_inputs,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(
            nn.Dropout(0.1),
            LinearBlock(num_inputs, 500, 0.2),
            LinearBlock(500, 500, 0.2),
            LinearBlock(500, num_pred_classes, 0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x.view(x.shape[0], -1))


class LinearBlock(nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 dropout: float) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
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


if __name__=='__main__':
    from tslearn.datasets import UCR_UEA_datasets
    import pickle
    dataset='ElectricDevices'
    train_x,train_y, test_x, test_y=UCR_UEA_datasets().load_dataset(dataset)
    train_x = train_x.reshape(-1,  train_x.shape[-2],1)
    test_x = test_x.reshape(-1, test_x.shape[-2],1)
    enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
    train_y=enc1.transform(train_y.reshape(-1,1))
    test_y=enc1.transform(test_y.reshape(-1,1))

    full_x= np.vstack([train_x, test_x])
    full_y= np.vstack([train_y, test_y])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( full_x, full_y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.33, random_state=42)
    n_pred_classes =train_y.shape[1]
    train_dataset = UCRDataset(X_train.astype(np.float64),y_train.astype(np.int64))
    test_dataset = UCRDataset(X_test.astype(np.float64),y_test.astype(np.int64))
    val_dataset = UCRDataset(X_val.astype(np.float64),y_val.astype(np.int64))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=256,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=256,shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)
    model = LinearBaseline(train_x.shape[-2],y_train.shape[-1])
    fit(model,train_loader, test_loader)
    torch.save(model.state_dict(), f'./models/{dataset}/FeedForward')

 


