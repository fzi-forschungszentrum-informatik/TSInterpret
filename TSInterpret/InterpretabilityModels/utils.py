from scipy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq
import numpy as np
import pandas as pd 
import random
import torch
from pyts.utils import windowed_view

from  sklearn.preprocessing import minmax_scale

from deprecated import deprecated




class functionCall_wrapper():
    def __init__(self,func,mode) -> None:
        self.func = func
        self.mode=mode
    def predict(self,item):
        if self.mode == 'time' :
            item=item.reshape(item.shape[0],item.shape[2],item.shape[1])
        if self.mode == 'feat' :
            item=item.reshape(item.shape[0],item.shape[1],item.shape[2])
        return self.func(item)
@deprecated
class torch_wrapper():
    def __init__(self,model,mode) -> None:
        self.model = model
        self.mode=mode
    def predict(self,item):
        '''Wrapper function for torch models.'''
        item = np.array(item.tolist(), dtype=np.float64)
        if self.mode == 'feat' :
            _ind=torch.from_numpy(item.reshape(-1,item.shape[-2],item.shape[-1]))
        if self.mode == 'time' :
            _ind=torch.from_numpy(item.reshape(-1,item.shape[-1],item.shape[-2]))
        out=self.model(_ind.float())
        y_pred = torch.nn.functional.softmax(out).detach().numpy()
        return y_pred
@deprecated
class tensorflow_wrapper():
    '''Wrapper function for tensorflow models.'''
    def __init__(self,model,mode) -> None:
        self.model = model
        self.mode = mode

    def predict(self,item):
        print(item.shape)
        print(self.mode)
        if self.mode == 'time' :
            print('Time')
            item=item.reshape(item.shape[0],item.shape[2],item.shape[1])
        if self.mode == 'feat' :
            print('Feat')
            item=item.reshape(item.shape[0],item.shape[1],item.shape[2])
        print(item.shape)
        out=self.model.predict(item)
        return out
@deprecated
class sklearn_wrapper():
    '''Wrapper function for sklearn models.'''

    def __init__(self,model,mode) -> None:
        self.model = model
        self.mode = mode

    def predict(self,item):
        if self.mode == 'time' :
            item=item.reshape(item.shape[0],item.shape[2],item.shape[1])
        if self.mode == 'feat' :
            item=item.reshape(item.shape[0],item.shape[1],item.shape[2])
        print('yU Do This')
        out=self.model.predict_proba(item)
        return out

def fourier_transform(timeseries,reference_set=None, return_fourier = False):
    '''
    Return the Fourier Transformation of a Time Series.

    Parameters
    ----------
    timeseries: np.array
        timeseries to be transformed
    reference_set: np.array 
        Reference set of size (#items, #MVariate, Time)
    return_fourier: bool
        True if only fourier transformed data is supposed to be returned, else manipulated timeseries is returned

    Returns
    ----------
    pertubed_transformed_timeseries: np.array
            has size (#MVariate, Time), manipulated time series
    fourier_timeseries: np.array
        Fourier Transformed Timeseries
    fourier_reference_set: np.array
        Fourier Transformed Timeseries
    


    TODO: 
    * Number of Pertued Series currently only returns one 
    '''
    fourier_timeseries = rfft(np.array(timeseries)) 
    
    if reference_set!=None:
        fourier_reference_set = rfft(np.array(reference_set))

    if return_fourier:
        return fourier_timeseries, fourier_reference_set
    
    #Manipulate Series 
    len_fourier = fourier_timeseries.shape[-1] #lentgh of fourier
    
    #Define variables
    length = 1
    num_slices = 1
    #Set up dataframe for slices with start and end value
    slices_start_end_value = pd.DataFrame(columns= ['Slice_number', 'Start', 'End'])
    #Include the first fourier band which should not be perturbed
    new_row = {'Slice_number': 0, 'Start':0, 'End':1}
    #append row to the dataframe
    slices_start_end_value = slices_start_end_value.append(new_row, ignore_index=True)
    #Get start and end values of slices and number slices with quadratic scaling
    start_idx = length
    end_idx = length
    while length < len_fourier:
        start_idx = length   #Start value
        end_idx = start_idx + num_slices**2 #End value
        end_idx = min(end_idx, len_fourier)
        
        new_row = {'Slice_number': num_slices, 'Start':start_idx, 'End':end_idx}
        #append row to the dataframe
        slices_start_end_value = slices_start_end_value.append(new_row, ignore_index=True)
        
        length = length + end_idx - start_idx
        num_slices = num_slices + 1
           
    tmp_fourier_series = np.array(fourier_timeseries.copy()) # timeseries.copy()
    max_row_idx = fourier_reference_set.shape[-1] 
    rand_idx = np.random.randint(0, max_row_idx)
    idx=random.randint(0, len (slices_start_end_value))
    start_idx = slices_start_end_value['Start'][idx]
    end_idx = slices_start_end_value['End'][idx]
    tmp_fourier_series[start_idx:end_idx] = fourier_reference_set[rand_idx, start_idx:end_idx].copy()
    perturbed_fourier_retransform = irfft(tmp_fourier_series)

    return perturbed_fourier_retransform

def authentic_opposing_information(timeseries, reference_set, window_size):
    '''
    Return authentic opossing information time series.

    Parameters
    ----------
    timeseries: np.array
        timeseries to be transformed
    reference_set: np.array 
        Reference set of size (#items, #MVariate, Time)
    window_size: int
        Window size to be oberated on 

    Returns
    ----------
    ind1: np.array
            manipulated time series
    

    TODO: 
    * MultiVariate TimeSeries
    * Cope with more than one change ? 
    '''
    window=window_size
    shape= np.array(timeseries).shape[-1]
    sample_series= random.choice(reference_set)
    if (shape/window).is_integer():
        ind1 = windowed_view(np.array(timeseries).reshape(1,-1), window, window_step= window)[0]
        sample_series=windowed_view(sample_series.reshape(1,-1), window, window_step= window)[0]
    else: 
        shape_new = window*(int(shape/window)+1)
        padded= np.zeros((1,shape_new))
        sample_padded= np.zeros((1,shape_new))
        padded[0, :shape]=np.array(timeseries).reshape(1,-1)
        sample_padded[0, :shape]=sample_series.reshape(1,-1)
        ind1 =windowed_view(np.array(padded).reshape(1,-1), window, window_step= window)[0]
        sample_series=windowed_view(sample_padded, window, window_step= window)[0]#sample_series.reshape(-1,window)

    index= random.randint(0,len(ind1)-1)
    ind1[index]=sample_series[index]
    new_shape = ind1.reshape(1,-1).shape[1]
    if new_shape>shape:
        diff= shape_new-shape 
        ind1= np.array(ind1).reshape(-1)[0:shape_new-diff]
       
    ind1=ind1.reshape(1,-1)
    return ind1 

def perturb_global_mean(m, dataframe_set, start_idx, end_idx):
    """
    Perturbes timeseries with global mean

    Parameters
    ----------
    m : Series
        Current timeseries.
    dataframe_set : Dataframe
        Test set.
    start_idx : Integer
        Start index of the current slice.
    end_idx : Integer
        End index of the current slice.

    Returns
    -------
    None.

    """
    
    m[start_idx:end_idx] = dataframe_set.mean() #Mean of all points in test set
    return m
    
def perturb_local_mean(m, dataframe_set, start_idx, end_idx):
    """
    Perturbes timeseries with local mean

    Parameters
    ----------
    m : Series
        Current timeseries.
    dataframe_set : Dataframe
        Test set.
    start_idx : Integer
        Start index of the current slice.
    end_idx : Integer
        End index of the current slice.

    Returns
    -------
    None.

    """
    m[start_idx:end_idx] = dataframe_set[:, start_idx: end_idx].mean() #Mean of all points in test set in that slice
    return m
    
def perturb_local_noise(m, dataframe_set, start_idx, end_idx):
    """
    Perturbes timeseries with local noise

    Parameters
    ----------
    m : Series
        Current timeseries.
    dataframe_set : Dataframe
        Test set.
    start_idx : Integer
        Start index of the current slice.
    end_idx : Integer
        End index of the current slice.

    Returns
    -------
    None.

    """
    
    mean = dataframe_set[:, start_idx: end_idx].mean() #Mean of points in that slice of all test sets
    std = dataframe_set[:, start_idx: end_idx].std() #Standard deviation of that points
    m[start_idx:end_idx] = np.random.normal(mean, std,end_idx - start_idx) #Draw points in this slice of normal distribution with mean and std
    #print(m)
    return m
    
def perturb_global_noise(m, dataframe_set, start_idx, end_idx):
    """
    Perturbes timeseries with global noise

    Parameters
    ----------
    m : Series
        Current timeseries.
    dataframe_set : Dataframe
        Test set.
    start_idx : Integer
        Start index of the current slice.
    end_idx : Integer
        End index of the current slice.

    Returns
    -------
    None.

    """
    
    mean = dataframe_set.mean() #Mean of all points in test set
    std = dataframe_set.std() #Standard deviation of all points in test set
    m[start_idx:end_idx] = np.random.normal(mean, std,
                                             end_idx - start_idx) # Draw points of normal distribution with mean and std
    return m

def perturb_authentic_mean(m, dataframe_set, start_idx, end_idx, rand_idx):
    """
    Perturbes timeseries with authentic mean

    Parameters
    ----------
    m : TYPE
        DESCRIPTION.
    dataframe_set : TYPE
        DESCRIPTION.
    start_idx : TYPE
        DESCRIPTION.
    end_idx : TYPE
        DESCRIPTION.
    rand_idx : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #Copy slice of other timeseries
    m[start_idx:end_idx] = dataframe_set[rand_idx, start_idx:end_idx].copy()   
    return m
