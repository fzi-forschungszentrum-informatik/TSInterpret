import re
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
import numpy as np 
from  sklearn import preprocessing
import torch
from torch.autograd import Variable 
import seaborn as sns
import matplotlib.pyplot as plt 
from TSInterpret.InterpretabilityModels.Saliency.Saliency_Base import Saliency

class Saliency_PTY(Saliency):
    '''
    '''
    def __init__(self, model, NumTimeSteps, NumFeatures, method='GRAD',mode='time',backend= 'torch',device='cpu') -> None:
        '''
        '''
        super().__init__(model, NumTimeSteps, NumFeatures, method,mode)
        self.method = method
        if method == 'GRAD':
            self.Grad = Saliency(model)
        elif method == 'IG':
            self.Grad= IntegratedGradients(model)
        elif method == 'GS':
            self.Grad= GradientShap(model)
        elif method == 'DL':
            self.Grad = DeepLift(model)
        elif method == 'DLS':
            self.Grad = DeepLiftShap(model)
        elif method == 'SG':
            Grad_ = Saliency(model)
            self.Grad = NoiseTunnel(Grad_)
        elif method == 'SVS':
            self.Grad = ShapleyValueSampling(model)
        elif method == 'FP':
            self.Grad = FeaturePermutation(model)
        elif method == 'FA':
            self.Grad = FeatureAblation(model)
        elif method == 'FO':
            self.Grad = Occlusion(model)
        if backend == 'torch':
            self.device='cpu'

    def explain(self,item,labels, TSR = True):
        #TODO differntiat between CNN and LSTM 
        mask=np.zeros((self.NumTimeSteps, self.NumFeatures),dtype=int)
        featureMask=np.zeros((self.NumTimeSteps, self.NumFeatures),dtype=int)
        print('feature mask', featureMask.shape)
        print('time mask', mask.shape)
        for i in  range (self.NumTimeSteps):
            mask[i,:]=i
        rescaledGrad= np.zeros((item.shape))
        #for i in  range (NumTimeSteps):
        #    featureMask[:,i]=i
        idx=0
        #for i,  samples  in enumerate(item).iterrows():

            #print('[{}/{}] {} {} model accuracy {:.2f}'\
            #                        .format(i,len(test_loader), models[m], args.DataName, Test_Acc))
        item = np.array(item.tolist(), dtype=np.float64)
        input=torch.from_numpy(item)

        input = input.reshape(-1, self.NumTimeSteps, self.NumFeatures).to(self.device)
        input = Variable(input,  volatile=False, requires_grad=True)

        batch_size = input.shape[0]
       
        inputMask= np.zeros((input.shape))
        inputMask[:,:,:]=mask
        inputMask =torch.from_numpy(inputMask).to(self.device)
        mask_single= torch.from_numpy(mask).to(self.device)
        mask_single=mask_single.reshape(1,self.NumTimeSteps, self.NumFeatures).to(self.device)
        #TODO what to do with labels 
        #labels=torch.tensor(labels.int().tolist()).to(self.device)
        input=input.reshape(-1, self.NumFeatures, self.NumTimeSteps)
        baseline_single=torch.from_numpy(np.random.random(input.shape)).float().to(self.device)
        baseline_multiple=torch.from_numpy(np.random.random((input.shape[0]*5,input.shape[1],input.shape[2]))).float().to(self.device)
        input = input.float()
        base=None
        has_sliding_window = None
        if(self.method=='GRAD'):
            attributions = self.Grad.attribute(input,target=labels)
        elif(self.method == 'IG'):
            base=baseline_single
            attributions = self.Grad.attribute(input, baselines=baseline_single,target=labels)
        elif(self.method=='DL'):
            base=baseline_single
            attributions = self.Grad.attribute(input,baselines=baseline_single,target=labels)
        elif(self.method=='GS'):
            base=baseline_multiple
            attributions = self.Grad.attribute(input, baselines=baseline_multiple, stdevs=0.09, target=labels)
        elif(self.method=='DLS'):
            base=baseline_multiple
            attributions = self.Grad.attribute(input, baselines=baseline_multiple, target=labels)
        elif(self.method=='SG'):
            attributions = self.Grad.attribute(input,target=labels)
        elif(self.method=='SVS'):
            base=baseline_single
            attributions = self.Grad.attribute(input, baselines=baseline_single, target=labels, feature_mask=inputMask)
        elif(self.method=='FP'):
            attributions = self.Grad.attribute(input, target=labels, perturbations_per_eval= input.shape[0],feature_mask=mask_single)
        elif(self.method=='FA'):
            attributions = self.Grad.attribute(input, target=labels)
                                            # perturbations_per_eval= input.shape[0],\
                                            # feature_mask=mask_single)
        elif(self.method=='FO'):
            #TODO Is this correct ? 
            base=baseline_single
            has_sliding_window = (1,self.NumFeatures)
            attributions = self.Grad.attribute(input,  sliding_window_shapes=(1,self.NumFeatures),target=labels,   baselines=baseline_single)

        if TSR:
            print('TSR is set true')
            #if self.mode =='feat':
            #    help = self.NumFeatures
            #    self.NumFeatures=self.NumTimeSteps
            #    self.NumTimeSteps=help
            #if self.mode =='time':
            #help = self.NumFeatures
            #self.NumFeatures=self.NumTimeSteps
            #self.NumTimeSteps=help
            TSR_attributions = self._getTwoStepRescaling(input, labels,hasBaseline=base,hasSliding_window_shapes=has_sliding_window)
            TSR_saliency=self._givenAttGetRescaledSaliency(TSR_attributions,isTensor=False)
            return TSR_saliency
        else:
            rescaledGrad[idx:idx+batch_size,:,:]=self._givenAttGetRescaledSaliency(attributions)
            return rescaledGrad

    def _getTwoStepRescaling(self,input, TestingLabel,hasBaseline=None,hasFeatureMask=None,hasSliding_window_shapes=None):
        '''From https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark/blob/main/MNIST%20Experiments/Scripts/interpret.py'''
        sequence_length=self.NumTimeSteps
        input_size=self.NumFeatures
        assignment=input[0,0,0]
        timeGrad=np.zeros((1,sequence_length))
        print('sequence length',sequence_length)
        inputGrad=np.zeros((input_size,1))
        print('inpu size',input_size)
        newGrad=np.zeros((input_size, sequence_length))
        if(hasBaseline==None):  
            ActualGrad = self.Grad.attribute(input,target=TestingLabel).data.cpu().numpy()
        else:
            if(hasFeatureMask!=None):
                print('hasFeatureMask')
                ActualGrad = self.Grad.attribute(input,baselines=hasBaseline, target=TestingLabel,feature_mask=hasFeatureMask).data.cpu().numpy()    
            elif(hasSliding_window_shapes!=None):
                print('HasSlidingWindow')
                ActualGrad = self.Grad.attribute(input,sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
            else:
                print('Else')
                ActualGrad = self.Grad.attribute(input,baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
                print(ActualGrad.shape)
        for t in range(sequence_length):
            newInput = input.clone()
            print(newInput.shape)
            newInput[:,:,t]=assignment

            if(hasBaseline==None):  
                timeGrad_perTime = self.Grad.attribute(newInput,target=TestingLabel).data.cpu().numpy()
            else:
                if(hasFeatureMask!=None):
                    timeGrad_perTime = self.Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel,feature_mask=hasFeatureMask).data.cpu().numpy()    
                elif(hasSliding_window_shapes!=None):
                    timeGrad_perTime = self.Grad.attribute(newInput,sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
                else:
                    timeGrad_perTime = self.Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()


            timeGrad_perTime= np.absolute(ActualGrad - timeGrad_perTime)
            timeGrad[:,t] = np.sum(timeGrad_perTime)



        timeContibution=preprocessing.minmax_scale(timeGrad, axis=1)
        meanTime = np.quantile(timeContibution, .55)    

    
        for t in range(sequence_length):
            if(timeContibution[0,t]>meanTime):
                for c in range(input_size):
                    newInput = input.clone()
                    newInput[:,c,t]=assignment

                    if(hasBaseline==None):  
                        inputGrad_perInput = self.Grad.attribute(newInput,target=TestingLabel).data.cpu().numpy()
                    else:
                        if(hasFeatureMask!=None):
                            inputGrad_perInput = self.Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel,feature_mask=hasFeatureMask).data.cpu().numpy()    
                        elif(hasSliding_window_shapes!=None):
                            inputGrad_perInput = self.Grad.attribute(newInput,sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
                        else:
                            inputGrad_perInput = self.Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()



                    inputGrad_perInput=np.absolute(ActualGrad - inputGrad_perInput)
                    inputGrad[c,:] = np.sum(inputGrad_perInput)
                    # print(t,c,np.sum(inputGrad_perInput),np.sum(input.data.cpu().numpy()))
                # featureContibution=inputGrad
                featureContibution= preprocessing.minmax_scale(inputGrad, axis=0)
            else:
                featureContibution=np.ones((input_size,1))*0.1
       
            # meanFeature=np.mean(featureContibution, axis=0)
            # for c in range(input_size): 
            #     if(featureContibution[c,0]<=meanFeature):
            #         featureContibution[c,0]=0
            for c in range(input_size):

                newGrad [c,t]= timeContibution[0,t]*featureContibution[c,0]
           # if(newGrad [c,t]==0):
           #  print(timeContibution[0,t],featureContibution[c,0])
        return newGrad

    def _givenAttGetRescaledSaliency(self,attributions,isTensor=True):
        if(isTensor):
            saliency = np.absolute(attributions.data.cpu().numpy())
        else:
            saliency = np.absolute(attributions)
        saliency=saliency.reshape(-1,self.NumTimeSteps*self.NumFeatures)
        rescaledSaliency=preprocessing.minmax_scale(saliency,axis=1)
        rescaledSaliency=rescaledSaliency.reshape(attributions.shape)
        return rescaledSaliency

