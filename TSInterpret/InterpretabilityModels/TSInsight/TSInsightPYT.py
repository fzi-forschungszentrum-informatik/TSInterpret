from pyexpat import model
from tkinter import N
from typing import Tuple
from zmq import device
from TSInterpret.InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
from TSInterpret.InterpretabilityModels.TSInsight.TSInsight import TSInsight
from TSInterpret.InterpretabilityModels.TSInsight.PYT_AE.RecurrentAutoencoder_PYT import RecurrentAutoencoder
from TSInterpret.InterpretabilityModels.TSInsight.PYT_AE.CNN_PYT import ConvAutoencoder
from TSInterpret.InterpretabilityModels.TSInsight.PYT_AE.Vanilla_PYT import Vanilla_Autoencoder
from ClassificationModels.CNN_T import UCRDataset
import torch.nn.functional as F
from torch.autograd import Variable
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from  captum.attr import InputXGradient
from captum.attr import (
    Saliency
)

class wrap_ae(nn.Module):
    def __init__(self,model,shape_ae,shape) -> None:
        super(wrap_ae, self).__init__()
        self.model = model 
        self.shape_ae = shape_ae
        self.shape=shape

    def forward(self,x):
        x=x.reshape(-1,self.shape_ae[0],self.shape_ae[1])
        x=self.model(x)
        x=x.reshape(-1, self.shape[0],self.shape[1])
        return x

class wrap_model(nn.Module):
    def __init__(self,model,shape) -> None:
        super(wrap_model, self).__init__()
        self.model = model 
        self.shape=shape

    def forward(self,x):
        x=x.reshape(-1, self.shape[0],self.shape[1])
        res=self.model(x)
        return res 


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
    def __init__(self, mlmodel,shape,data, test_data=None, backend='PYT',autoencoder = None, device = 'cpu',loss_fn=nn.MSELoss(), lr=0.001,**kwargs):
        
        super().__init__(mlmodel,shape, 'time', backend)
        self.device=device
        self.saliency=Saliency(mlmodel)
        self.loss_fn= loss_fn
        self.lr=lr
        self.shape_ae=shape
        # Check if classification model is cnn: 
        self.cnn_model = self._check_cnn(mlmodel)
        if self.cnn_model:
            self.mode='feat'
        self.cnn_ae = False
        print(type(autoencoder))
        if isinstance(autoencoder,str):
            if autoencoder == 'cnn':
                self.cnn_ae=True
        else:
            print('Check cnn')
            self.cnn_ae=self._check_cnn(autoencoder)
        print('cnn ae flag',  self.cnn_ae)
        print('cnn_model', self.cnn_model)
        #if not cnn_ae and cnn_model:
        #    print('Model Shape',shape)
        #    self.model=wrap_model(model,shape)
        #    self.shape_ae=(self.shape_ae[-1], self.shape_ae[-2])
        

        if isinstance(data, Tuple):
            #Only Builds Datasets, no manipulation
            print('INFO - Provided Datasets are Tuples. Create a Default DataLoader.')
            train_x,train_y= data
            test_x,test_y=test_data
            train_dataset = UCRDataset(train_x.astype(np.float64),train_y.astype(np.int64))
            test_dataset = UCRDataset(test_x.astype(np.float64),test_y.astype(np.int64))
            data = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)
            test_data  = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
            self.shape=(train_x.shape[1],train_x.shape[2])
        self.flatten=False
        if autoencoder == None:
            self.flatten= True
            #CNN not an issue as autoencoder is flattend ? 
            self.autoencoder=Vanilla_Autoencoder(shape[0]*shape[1])
            self.autoencoder =self.autoencoder.to(self.device)
            self.autoencoder=self._train(data,test_data,**kwargs)
        elif autoencoder=='reccurent':
            self.flatten= False
            print('Shape_ae',self.shape_ae)
            if self.cnn_model:
                self.shape_ae=(shape[1],shape[0])
                self.autoencoder=RecurrentAutoencoder(self.shape_ae[0], self.shape_ae[1], 50)
            else:
                self.autoencoder = RecurrentAutoencoder(self.shape_ae[0], self.shape_ae[1], 50) # 128
            self.autoencoder = self.autoencoder.to(device)
            self.autoencoder=self._train(data,test_data,**kwargs)
        elif autoencoder=='cnn':
            self.flatten= False
            if not self.cnn_model:
                self.shape_ae=(train_x.shape[2],train_x.shape[1])
              
                self.autoencoder =ConvAutoencoder(self.shape_ae[0]) # 128
                
            self.autoencoder = self.autoencoder.to(device)
            self.autoencoder=self._train(data,test_data,**kwargs)

        else: 
            self.flatten= False
            if self.cnn_ae and not self.cnn_model:
                self.shape_ae=(train_x.shape[2],train_x.shape[1])
                self.autoencoder=autoencoder
            elif not self.cnn_ae and self.cnn_model:
                self.shape_ae=(train_x.shape[2],train_x.shape[1])
                self.autoencoder=autoencoder
            else:
                self.autoencoder = autoencoder
        
        
        self._fine_tuning(data,test_data,**kwargs)
    
    def _check_cnn(self,model):
        for layer in model.modules():#.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv2d):
                return True 
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d):
                return True
        return False



    def explain(self, item):
        """
        Explains an instance. 
        Attributes:
        item np.array: Instance to be explained
        """
        item = item.reshape(-1,self.shape_ae[0],self.shape_ae[1])
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


    def _train(self, dataloader, test_dataloader=None, epochs=1000, loss_fn=nn.MSELoss(), lr=0.001, patience=20,weight_decay=1e-5,contractive_autoencoder=False): # reduction_factor=0.9, reduction_tolerance=4, patience=20, lam = 0.2,l2 = True):
        # Pretrains the Autoencoder, if necessary 
        optim = torch.optim.Adam(self.autoencoder.parameters(),lr=lr,weight_decay=weight_decay)

        train_losses = []
        validation_losses = []
        best_model=None
        best_loss = 1000
        trigger_times=0
        self.autoencoder.train()
        for i in range(epochs):
            for batch,_ in dataloader:
                batch = batch.to(self.device).reshape(-1,self.shape_ae[0],self.shape_ae[1])
            
                if self.flatten:
                    batch = batch.view(batch.size(0), self.shape_ae[0]*self.shape_ae[1])
                optim.zero_grad()
                reconstruction= self.autoencoder(batch.float())
                loss = loss_fn(batch.float(), reconstruction )
                loss.backward()
                optim.step()
                train_losses.append(loss.item())

            if test_dataloader!= None:
                self._evaluate(validation_losses, test_dataloader)
                print(f'Epoch: {i}, '
                  f'Train loss: {round(train_losses[-1], 3)}, '+f'Validation loss: {round(validation_losses[-1], 3)}')
            else:
                print(f'Epoch: {i}, '
                  f'Train loss: {round(train_losses[-1], 3)}')
            
            if validation_losses[-1] > best_loss:
                trigger_times += 1
            else:
                best_model=self.autoencoder
                best_loss=validation_losses[-1]
                trigger_times =0
            if trigger_times >= patience:
                self.autoencoder=best_model
                return self.autoencoder
        return self.autoencoder


    def _evaluate(self,losses, dataloader):
        loss = self.calculate_loss( dataloader)
        losses.append(loss)

    def calculate_loss(self, dataloader, loss_fn=nn.MSELoss()):
        losses = []
        model= self.autoencoder
        for batch, labels in dataloader:
            batch = batch.to(self.device).reshape(-1,self.shape_ae[0],self.shape_ae[1])
            labels = labels.to(self.device)
        
            if self.flatten:
                batch = batch.view(batch.size(0),self.shape_ae[0]*self.shape_ae[1])
            reconstruction= self.autoencoder(batch.float()).reshape(-1,self.shape_ae[0],self.shape_ae[1])
            loss = loss_fn(batch.float(), reconstruction)
            
            losses.append(loss)

        return (sum(losses)/len(losses)).item() # calculate mean
    def _hyperparameter(self,batch,target):
        gamma=[]
        ßeta=[]
        self.saliency=Saliency(self.model)
        for x,y in zip(batch, target):
            x=x.reshape(-1,self.shape_ae[0],self.shape_ae[1]).float()
            sal= self.saliency.attribute(x,np.argmax(y))
            sal=sal[0].detach().numpy()
            I= (sal- np.ones_like(sal)*np.min(sal))/( np.ones_like(sal)*np.max(sal)- np.ones_like(sal)*np.min(sal))
            gamma.append(I)
            ßeta.append(np.ones_like(I)-I)
            #print(ßeta)
        return torch.from_numpy(np.array(ßeta)).float(), torch.from_numpy(np.array(gamma)).float()
    
    def _fine_tuning(self, dataloader,test_data, epochs=50, lr=0.0001, patience=10,l1=True,ß=0.0001,om=4.0,lam=0.2, C=10, self_tune = False):
        # Parameters are set according to the Paper. 
        self.model.train()
        self.autoencoder.train()
        best_loss=1000
        loss_fn1=nn.CrossEntropyLoss()
        loss_fn2=nn.MSELoss()
        optim=torch.optim.Adam(self.autoencoder.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.9, patience=4)
        trigger_times=0
        best_model=self.autoencoder
        train_losses = []
        validation_losses = []
        for i in range(epochs):
            for batch,labels in dataloader:
                batch = batch.to(self.device).reshape(-1,self.shape_ae[0],self.shape_ae[1])
                #batch = batch.to(self.device)
            
                if self.flatten:
                    batch = batch.view(batch.size(0), self.shape_ae[0]*self.shape_ae[1])

                optim.zero_grad()
                batch = batch.to(self.device)
                labels = labels.to(self.device)
        
                if self.flatten:
                    batch = batch.view(batch.size(0),self.shape_ae[0]*self.shape_ae[1])
                l2_reg = torch.tensor(0.)
                for param in self.autoencoder.parameters():
                    l2_reg += torch.norm(param)
                

                reconstruction= self.autoencoder(batch.float()).reshape(-1,self.shape_ae[0],self.shape_ae[1])
                if self_tune:
                    #TODO Add Params again 
                    ß,gamma=self._hyperparameter(batch,labels)
                    #self.model.eval()
                    loss1 = loss_fn1(self.model( reconstruction.reshape(-1,self.shape[0],self.shape[1])),labels.float())
                    loss2=torch.sum(torch.pow((batch.float()- reconstruction)*gamma,2)) 
                    loss3= C*torch.sum(torch.abs( reconstruction*ß)).float()
                    l2= lam* l2_reg
                    loss= loss1 +loss2+ loss3  + l2 
                else:
                    loss1 = loss_fn1(self.model( reconstruction.reshape(-1,self.shape[0],self.shape[1])),labels.float())
                    loss2=om*loss_fn2(batch.float(), reconstruction)
                    loss3= ß* torch.sum(torch.abs( reconstruction)).float() 
                    l2= lam* l2_reg
                    loss = loss1+ loss2+loss3+l2
                   
                loss.backward()
                optim.step()
                train_losses.append(loss.item())
            self.model.eval()
            self.autoencoder.eval()
            for batch, labels in test_data:
                
                reconstruction= self.autoencoder(batch.float().reshape(-1,self.shape_ae[0],self.shape_ae[1]))
                if self_tune:
                    ß,gamma=self._hyperparameter(batch,labels)
                    loss1_val = loss_fn1(self.model( reconstruction.reshape(-1,self.shape[0],self.shape[1])),labels.float())
                    loss2_val=torch.sum(torch.pow((batch.float()- reconstruction)*gamma,2)) 
                    loss3_val= C*torch.sum(torch.abs( reconstruction*ß)).float() 
                    l2_val= lam* l2_reg
                    loss_val= loss1_val +loss2_val+ loss3_val  + l2_val
                else:
                    loss1_val = loss_fn1(self.model( reconstruction.reshape(-1,self.shape[0],self.shape[1])),labels.float())
                    loss2_val=om*loss_fn2(batch.float(), reconstruction) 
                    loss3_val= ß* torch.sum(torch.abs( reconstruction)).float()
                    l2_val= lam* l2_reg
                    loss_val = loss1_val+ loss2_val+loss3_val+l2_val
                val_loss =  loss_val 
                validation_losses.append(val_loss.item())
  
            print(f'Epoch: {i}, '
                  f'Fine Tune Loss: {round(train_losses[-1], 3)}, consits of {loss1}, {loss2}, {loss3}, {l2}')
            print(f'Epoch: {i}, '
                  f'Fine Validation Loss: {round(validation_losses[-1], 3)}, consits of {loss1_val}, {loss2_val}, {loss3_val}, {l2_val}')
                
            scheduler.step(val_loss)

            if val_loss > best_loss:
                trigger_times += 1
            else:
                best_model=self.autoencoder
                best_loss=val_loss
                trigger_times =0
            if trigger_times >= patience:
                self.autoencoder=best_model
                print('Early Stopping')
                return 
        self.autoencoder=best_model

        

