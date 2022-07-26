from zmq import device
from TSInterpret.InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
from TSInterpret.InterpretabilityModels.TSInsight.TSInsight import TSInsight
from TSInterpret.InterpretabilityModels.TSInsight.RecurrentAutoencoder_PYT import RecurrentAutoencoder
import torch.nn.functional as F
from torch.autograd import Variable
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from  captum.attr import InputXGradient
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Saliency,
    NoiseTunnel,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation,
    Occlusion

)
class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)

class Vanilla_Autoencoder(nn.Module):
    def __init__(self, input_size, hidden=10):
        super(Vanilla_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, hidden)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self,x):
        return self.encoder(x)

class TSInsightPYT(TSInsight):
    '''
        Instantiated TS Insight
        Attributes:
        mlmodel: Machine Learning Model to be explained.
        data: PYT data loader.
        data: PYT Test data loader.
        mode str: Second dimension is feature --> 'feat', is time --> 'time'.
        backend str: PYT or TF.
        autoencoder: None or instance of an implemented and traine AE. 
        device str: device to run optimization on.
        loss_fn func: Loss function instance for training AE, only if AE is None.
        lr float: learning rate for training AE, only if AE is None.
        
    '''
    def __init__(self, mlmodel,shape,data, test_data=None, mode='feat', backend='PYT',autoencoder = None, device = 'cpu',loss_fn=nn.MSELoss(), lr=0.001,**kwargs):
        super().__init__(mlmodel,shape, mode, backend)
        self.device=device
        self.saliency=Saliency(mlmodel)
        self.loss_fn= loss_fn
        self.lr=lr
        if autoencoder == None:
            self.autoencoder=Vanilla_Autoencoder(shape[0]*shape[1])
            self.autoencoder =self.autoencoder.to(self.device)
            self.autoencoder=self._train(data,test_data,**kwargs)
        elif autoencoder=='reccurent':
            self.autoencoder = RecurrentAutoencoder(shape[0], shape[1], 128)
            self.autoencoder = self.autoencoder.to(device)
            self.autoencoder=self._train(data,test_data,**kwargs)

        else: 
            self.autoencoder=autoencoder
        self._fine_tuning(data,**kwargs)

    def explain(self, item, flatten=True):
        """
        Explains an instance. 
        Attributes:

        item np.array: Instance to be explained
        flatten bool: Was autoencoder trained with flattend input.
        """
        if flatten:
            item = item.reshape(-1,self.shape[0],self.shape[1])
        output= self.autoencoder(torch.from_numpy(item).float()).detach().numpy()
        return output
    
    def l1_penalty(self,batch):
        layers= list(self.autoencoder.children())
        loss = 0
        values = batch.float()
        for i in range(len(layers)):
            values = F.relu((layers[i](values)))
            loss += torch.mean(torch.abs(values))
        return loss


    def _train(self, dataloader, test_dataloader=None, epochs=1000, flatten=True, loss_fn=nn.MSELoss(), lr=0.001, patience=20,weight_decay=1e-5,contractive_autoencoder=False): # reduction_factor=0.9, reduction_tolerance=4, patience=20, lam = 0.2,l2 = True):
        '''
        Train Autoencoder, no parameters on how to do that in paper. 
           
        '''
        #weight decay--> supposed to be L2 Norm at the end of formulation 
        optim = torch.optim.Adam(self.autoencoder.parameters(),lr=lr,weight_decay=weight_decay)
    
        train_losses = []
        validation_losses = []
        best_model=None
        best_loss = 1000
        trigger_times=0
        self.autoencoder.train()
        for i in range(epochs):
            for batch,_ in dataloader:
                batch = batch.to(self.device)
            
                if flatten:
                    batch = batch.view(batch.size(0), self.shape[0]*self.shape[1])

                optim.zero_grad()
                reconstruction= self.autoencoder(batch.float()).reshape(-1,self.shape[0],self.shape[1])
                loss = loss_fn(batch.float(), reconstruction )
                loss.backward()
                optim.step()
                train_losses.append(loss.item())

            if test_dataloader!= None:
                self._evaluate(validation_losses, test_dataloader, flatten)
                print(f'Epoch: {i}, '
                  f'Train loss: {round(train_losses[-1], 3)}, '+f'Validation loss: {round(validation_losses[-1], 3)}')
            else:
                print(f'Epoch: {i}, '
                  f'Train loss: {round(train_losses[-1], 3)}')
            
            if validation_losses[-1] > best_loss:
                trigger_times += 1
            else:
                print('Best Model is triggered')
                best_model=self.autoencoder
                best_loss=validation_losses[-1]
                trigger_times =0
            if trigger_times >= patience:
                self.autoencoder=best_model
                print('Early Stopping')
                return self.autoencoder
        return self.autoencoder


    def _evaluate(self,losses, dataloader, flatten=True):
        loss = self.calculate_loss( dataloader, flatten=flatten)
        losses.append(loss)

    def calculate_loss(self, dataloader, loss_fn=nn.MSELoss(), flatten=True):
        losses = []
        model= self.autoencoder
        for batch, labels in dataloader:
            batch = batch.to(self.device)
            labels = labels.to(self.device)
        
            if flatten:
                batch = batch.view(batch.size(0),self.shape[0]*self.shape[1])
            reconstruction= self.autoencoder(batch.float()).reshape(-1,self.shape[0],self.shape[1])
            loss = loss_fn(batch.float(), reconstruction)
            
            losses.append(loss)

        return (sum(losses)/len(losses)).item() # calculate mean
    def _hyperparameter(self,batch,target):
        '''
        #TODO check Calc
        #TODO More Efficiency?
        '''
        gamma=[]
        ßeta=[]
        for x,y in zip(batch, target):
            x=x.reshape(-1,self.shape[0],self.shape[1]).float()
            sal= self.saliency.attribute(x,np.argmax(y))
            sal=sal[0].detach().numpy()
            I= (sal- np.ones_like(sal)*np.min(sal))/( np.ones_like(sal)*np.max(sal)- np.ones_like(sal)*np.min(sal))
            gamma.append(I)
            ßeta.append(np.ones_like(I)-I)
        return torch.from_numpy(np.array(ßeta)).float(), torch.from_numpy(np.array(gamma)).float()
    
    def _fine_tuning(self, dataloader, epochs=50, flatten=True, lr=0.0001, patience=10,l1=True,ß=0.0001,om=4.0,lam=0.2, C=10, self_tune = True):
        #TODO Hyperparameter Tuning L1 = True, Learing Rate Schedulare  Learning rate reduction factor 0.9 Learning rate reduction tolerance 4
        best_loss=1000
        loss_fn1=nn.CrossEntropyLoss()
        loss_fn2=nn.MSELoss()
        optim = torch.optim.Adam(self.autoencoder.parameters(),lr=lr)
        trigger_times=0
        best_model=self.autoencoder
        train_losses = []
        validation_losses = []
        for i in range(epochs):
            for batch,labels in dataloader:
                batch = batch.to(self.device)
            
                if flatten:
                    batch = batch.view(batch.size(0), self.shape[0]*self.shape[1])

                optim.zero_grad()
                batch = batch.to(self.device)
                labels = labels.to(self.device)
        
                if flatten:
                    batch = batch.view(batch.size(0),self.shape[0]*self.shape[1])
                l1_pen=0
                if l1 :
                    l1_pen=self.l1_penalty(batch)
                l2_reg = torch.tensor(0.)
                for param in self.autoencoder.parameters():
                    l2_reg += torch.norm(param)
                

                reconstruction= self.autoencoder(batch.float()).reshape(-1,self.shape[0],self.shape[1])
                if self_tune:
                    ß,gamma=self._hyperparameter(batch,labels)
                    loss= loss_fn1(self.model(reconstruction),labels.float())+loss_fn2(batch.float()*gamma, reconstruction*gamma)+ C*torch.sum(torch.abs(self.autoencoder(batch.float())*ß)).float()  + lam* l2_reg#(torch.sum(self.autoencoder.encoder.weight.data)+torch.sum(self.autoencoder.decoder.weight.data)) 
                else: 
                    loss= loss_fn1(self.model(reconstruction),labels.float())+om*loss_fn2(batch.float(), reconstruction)+ ß* l1_pen  + lam* l2_reg#(torch.sum(self.autoencoder.encoder.weight.data)+torch.sum(self.autoencoder.decoder.weight.data)) 

                loss.backward()
                optim.step()
                train_losses.append(loss.item())
            print(f'Epoch: {i}, '
                  f'Fine Tune Loss: {round(train_losses[-1], 3)}')
            if loss > best_loss:
                trigger_times += 1
            else:
                best_model=self.autoencoder
                best_loss=loss
                trigger_times =0
            if trigger_times >= patience:
                self.autoencoder=best_model
                print('Early Stopping')
                return 
        self.autoencoder=best_model

        

