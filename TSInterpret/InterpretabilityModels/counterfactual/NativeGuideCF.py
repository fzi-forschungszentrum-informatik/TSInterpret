#TODO 
'''Implementation after Delaney et al . https://github.com/e-delaney/Instance-Based_CFE_TSC'''
from cProfile import label
import imp
from InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
from itertools import count
from operator import sub
from tslearn.neighbors import KNeighborsTimeSeries
from torchcam.methods import SmoothGradCAMpp, GradCAM
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.vanilla_gradients import VanillaGradients
from torchcam.methods import CAM
import numpy as np 
from pathlib import Path
import platform
import os 
import pandas as pd
from InterpretabilityModels.GradCam.GradCam_1D import grad_cam, GradCam1D
import numpy as np
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
import tensorflow as tf
from typing import Tuple
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from tslearn.barycenters import dtw_barycenter_averaging
#from InterpretabilityModels.utils import torch_wrapper, tensorflow_wrapper
from Models.PyTorchModel import PyTorchModel
from Models.TensorflowModel import TensorFlowModel
from Models.SklearnModel import SklearnModel
from InterpretabilityModels.counterfactual.CF import CF

class NativeGuideCF(CF):
    '''
    NUN_CF according to 
    Delaney, E., Greene, D., Keane, M.T.: Instance-Based Counterfactual Explanations
    for Time Series Classification. In: S ́anchez-Ruiz, A.A., Floyd, M.W. (eds.) Case-
    Based Reasoning Research and Development, vol. 12877, pp. 32–47. Springer
    International Publishing, Cham (2021), series Title: Lecture Notes in Computer
    Science     for both torch and tensorflow. 
    '''
    def __init__(self, model,shape, reference_set, backend='torch', mode ='feat') -> None:
        '''
        In this case differentiation between time & feat not necessary as implicitly given by CNN. Only works for CNNs due to the attribution methods.
        Args:
            model: classification model to explain
            shape: input shape
            reference set: reference set as tuple
            mode: model type either torch or tensorflow
        '''
        super().__init__(model,mode)
        #self.model=model
        self.backend=backend
        test_x,test_y=reference_set
        test_x=np.array(test_x,dtype=np.float32)
        self.mode=mode
        if mode == 'time':
            #Parse test data into (1, feat, time):
            self.ts_length= shape[-2]
            test_x=test_x.reshape(test_x.shape[0],test_x.shape[2],test_x.shape[1])
        elif mode == 'feat':
            self.ts_length= shape[-1]

        if backend == 'PYT':
            
            try:
                #self.cam_extractor =SmoothGradCAMpp(self.model,input_shape=(shape[1],shape[2]) )
                self.cam_extractor =CAM(self.model,input_shape=(shape[1],shape[2]) )
            except:
                print('GradCam Hook already registered')
            #out=self.model(torch.from_numpy(test_x.reshape(-1,1,test_x.shape[-1])))
            change=False
            if self.mode =='time':
                change= True
            self.predict=PyTorchModel(self.model,change).predict
            y_pred = np.argmax(self.predict(test_x), axis=1)
            
        elif backend== 'TF':
            self.cam_extractor =GradCam1D() #VanillaGradients()#GradCam1D()
            y_pred = np.argmax(self.model.predict(test_x.reshape(-1,self.ts_length,1)), axis=1)
            self.predict=TensorFlowModel(self.model,change=True).predict
            #Parse test data into torch format : 
            
        else: 
            print('Only Compatible with Tensorflow (TF) or Pytorch (PYT)!')

        self.reference_set=(test_x,y_pred)
        #Manipulate reference set replace original y with predicted y 
        
    def native_guide_retrieval(self,query, predicted_label, distance, n_neighbors):
        '''
        This gets the nearest unlike neighbors.
        Args:
            query (np.array): The instance to explain.
            predicted_label (np.array): Label of instance.
            reference_set (np.array): Set of addtional labeled data (could be training or test set)
            distance (str): 
            num_neighbors (int):number nearest neighbors to return
        Returns:
            [np.array]: Returns K_Nearest_Neighbors of input query with different classification label.
    
        '''
        if type(predicted_label) != int:
            predicted_label=np.argmax(predicted_label)
    
        x_train, y=self.reference_set
        if len(y.shape)==2:
            y=np.argmax(y,axis=1)
        ts_length=self.ts_length
        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)
        knn.fit(x_train[list(np.where(y != predicted_label))].reshape(-1,ts_length,1))
        dist,ind = knn.kneighbors(query.reshape(1,ts_length,1), return_distance=True)
        x_train.reshape(-1,1,ts_length)
        return dist[0],x_train[np.where(y != predicted_label)][ind[0]]

    def findSubarray(self, a, k): #used to find the maximum contigious subarray of length k in the explanation weight vector
    
        n = len(a)
    
        vec=[] 

        # Iterate to find all the sub-arrays 
        for i in range(n-k+1): 
            temp=[] 

            # Store the sub-array elements in the array 
            for j in range(i,i+k): 
                temp.append(a[j]) 

            # Push the vector in the container 
            vec.append(temp) 

        sum_arr = []
        for v in vec:
            sum_arr.append(np.sum(v))

        return (vec[np.argmax(sum_arr)])

    def counterfactual_generator_swap(self,instance, label,subarray_length=1,max_iter=500):
        print(label)
        _,nun=self.native_guide_retrieval(instance, label, 'euclidean', 1)
        if np.count_nonzero(nun.reshape(-1)-instance.reshape(-1))==0:
            print('Starting and nun are Identical !')

        test_x,test_y=self.reference_set
        train_x=test_x
      
        #Classify native guide
        #input_ = torch.from_numpy(np.array(nun)).float().reshape(-1,1,train_x.shape[-1])
        #out = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        individual = np.array(nun.tolist(), dtype=np.float64)
        out=self.predict(individual)
        print('Native Guide', out)
        if self.backend=='PYT': 
            training_weights = self.cam_extractor(out.squeeze(0).argmax().item(), out)[0].detach().numpy()
        elif self.backend=='TF':
            data = (instance.reshape(1,-1,1), None)
            #label= tf.Tensor([[0,0,0,0,1,0,0,0,0,0]],dtype=tf.int32)
            training_weights =self.cam_extractor.explain(data, self.model,class_index=label[0])# grad_cam(self.model, instance.reshape(1,-1,1))#self.cam_extractor.explain(data, self.model,class_index=label)#instance
        #Classify Original
        individual = np.array(instance.tolist(), dtype=np.float64)
        out=self.predict(individual)
        #input_ = torch.from_numpy(np.array(instance)).float().reshape(-1,1,train_x.shape[-1]) 
        #out = torch.nn.functional.softmax(self.model(input_)).detach().numpy()


        most_influencial_array=self.findSubarray((training_weights), subarray_length)
    
        starting_point = np.where(training_weights==most_influencial_array[0])[0][0]
        print('Starting Points', starting_point)
    
        X_example = instance.copy().reshape(1,-1)
        
        nun=nun.reshape(1,-1)
        print('Subarray Length', subarray_length)
        X_example[0,starting_point:subarray_length+starting_point] =nun[0,starting_point:subarray_length+starting_point]
        #np.concatenate(instance[:starting_point], nun[starting_point:subarray_length+starting_point], instance[subarray_length+starting_point:])
        #print(np.argmax(label,axis=0))
        individual = np.array(X_example.reshape(-1,1,train_x.shape[-1]).tolist(), dtype=np.float64)
        out=self.predict(individual)
        #input_ = torch.from_numpy(np.array(X_example)).float().reshape(-1,1,train_x.shape[-1]) 
        #out = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        prob_target =  out[0][label] #torch.nn.functional.softmax(model(torch.from_numpy(test_x))).detach().numpy()[0][y_pred[instance]]
        #print(out)
        print('Prob_target',prob_target)
        counter= 0
        while prob_target > 0.5 and counter <max_iter:
        
            subarray_length +=1        
            most_influencial_array=self.findSubarray((training_weights), subarray_length)
            starting_point = np.where(training_weights==most_influencial_array[0])[0][0]
            print('starting',starting_point)
            print(subarray_length)
            X_example = instance.copy().reshape(1,-1)
            X_example[:,starting_point:subarray_length+starting_point] =nun[:,starting_point:subarray_length+starting_point]
            print(np.count_nonzero(nun.reshape(-1)-X_example.reshape(-1))==0)
            individual = np.array(X_example.reshape(-1,1,train_x.shape[-1]).tolist(), dtype=np.float64)
            out=self.predict(individual)
            #input_ = torch.from_numpy(np.array(X_example)).float().reshape(-1,1,train_x.shape[-1]) #TODO nun or instance
            #out = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
            prob_target = out[0][label]
            #print(out)
            print('Prob_target',prob_target)
            counter=counter+1
            if counter==max_iter or subarray_length==self.ts_length:
                print('No Counterfactual found')
                return None, None
        
        return X_example,np.argmax(out,axis=1)[0]

    def instance_based_cf(self,query,label,target=None, distance='dtw', max_iter=500):
    
        d,nan=self.native_guide_retrieval(query, label, distance, 1)
        beta = 0
        #TODO reshape
        insample_cf = nan.reshape(1,1,-1)

        individual = np.array(query.tolist(), dtype=np.float64)

        output=self.predict(individual)
        #input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
        #output = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        pred_treshold = 0.5
        target = np.argsort(output)[0][-2:-1][0] 
        query=query.reshape(-1)
        insample_cf=insample_cf.reshape(-1)
        generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1-beta), beta]))
        generated_cf = generated_cf.reshape(1,1,-1)
        individual = np.array(generated_cf.tolist(), dtype=np.float64)
        prob_target=self.predict(individual)[0][target]
        #input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
        #prob_target = torch.nn.functional.softmax(self.model(input_)).detach().numpy()[0][target]
        counter=0

        while prob_target < pred_treshold and counter<max_iter:
            beta +=0.01 
            generated_cf= dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1-beta), beta]))
            generated_cf = generated_cf.reshape(1,1,-1)

            individual = np.array(generated_cf.tolist(), dtype=np.float64)
            #input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
            #prob_target = torch.nn.functional.softmax(self.model(input_)).detach().numpy()[0][target]
            prob_target=self.predict(individual)[0][target]

            counter=counter+1
        if counter==max_iter:
            print('No Counterfactual found')
            return None, None
    
        return generated_cf, target

    #TODO Output type 
    def explain(self, x: np.ndarray,  y: int, method = 'NUN_CF', distance_measure='dtw',n_neighbors=1,max_iter=500)-> Tuple[np.array, int]:
        ''''
        Explains a specific instance x 
        
        '''
        if self.mode =='time':
            x=x.reshape(x.shape[0], x.shape[2],x.shape[1])
            print(x.shape)
        if method=='NUN_CF':
            dis, item= self.native_guide_retrieval(x, y, distance_measure, n_neighbors)
            return item
        elif method=='dtw_bary_center':
            return self.instance_based_cf(x,y,distance_measure)
        elif method=='native_guide':
            distance_measure='euclidean'
            return  self.counterfactual_generator_swap(x, y,max_iter=max_iter)
        else: 
            print('Unknown Method selected.')

