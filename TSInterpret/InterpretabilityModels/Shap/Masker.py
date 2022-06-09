from InterpretabilityModels.utils import authentic_opposing_information, fourier_transform, perturb_local_noise
import shap
# TODO

from shap.maskers import Masker
import sklearn
from InterpretabilityModels.utils import * #(m, dataframe_set, start_idx, end_idx):
from shap._serializable import Serializer, Deserializer
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import random 
import math 
import pandas as pd
class TimeSeriesMasker(Masker):
    """ This masks out image regions with blurring or inpainting.
    TODO 
    * All Methods 
    * Visualization 
    """

    def __init__(self,mask_type,data,shape, max_samples=100, clustering=None, ):
        """ Build a Time Series Masker.
        Parameters
        ----------
        mask_value : np.array, "blur(kernel_xsize, kernel_xsize)", "inpaint_telea", or "inpaint_ns"
            The value used to mask hidden regions of the image.
        shape : None or tuple
            If the mask_value is an auto-generated masker instead of a dataset then the input
            image shape needs to be provided.
        """
  
      
        self.input_shape = shape
        self.mask_type =mask_type
        #self.input_mask_value = mask_value
        self.data = data 

        self.len_ts = shape[-1]*shape[-2]  #length of timeseries  
        num_slices = min(math.floor(self.len_ts/5), 30) #number slices
        self.values_per_slice = math.floor(self.len_ts / num_slices) #calculate values per slice for minimum num_slice slices
        num_slices = num_slices + math.ceil((self.len_ts - num_slices * self.values_per_slice)/ self.values_per_slice) #Calculate adapted number slices to divide the rest in equal parts
        #deact_per_sample = np.random.randint(1, num_slices + 1, num_samples - 1) #random draw of inactive slices per sample
        #perturbation_matrix = mask #np.ones((num_samples, num_slices)) #perturbation matrix with ones
        slices_start_end_value = pd.DataFrame(columns= ['Slice_number', 'Start', 'End'])
        for j in range(num_slices):
            start_idx = j * self.values_per_slice
            end_idx = start_idx + self.values_per_slice
            end_idx = min(end_idx, self.len_ts)
            new_row = {'Slice_number': j, 'Start':start_idx, 'End':end_idx}
            #append row to the dataframe
            slices_start_end_value = slices_start_end_value.append(new_row, ignore_index=True)

        self.info=slices_start_end_value
        self.mask=np.zeros(len(slices_start_end_value))
        if self.mask_type == 'local_noise':
            self.perturb =perturb_local_noise
        elif self.mask_type == 'global_noise':
            self.perturb =perturb_global_noise
        elif self.mask_type == 'local_mean':
            self.perturb =perturb_local_noise
        elif self.mask_type == 'global_mean':
            self.perturb =perturb_global_mean
        elif self.mask_type == 'fourier':
            self.perturb =fourier_transform
        elif self.mask_type == 'authentic':
            self.perturb =authentic_opposing_information
        

        # This is the shape of the masks we expect
        #self.shape = (1, np.prod(self.input_shape)) # the (1, ...) is because we only return a single masked sample to average over


    def __call__(self, mask,x):
        # unwrap single element lists (which are how single input models look in multi-input format)
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        # we preserve flattend inputs as flattened and full-shaped inputs as their original shape
        in_shape = x.shape
        if len(x.shape) > 1:
            x = x.ravel()

        if type(self.mask_type)==str:
            out=x
            idx=np.where(mask==True)
            if len(idx[0]) != 0:
                for item in idx:
                    #print(item[0])
                    row= self.info[(self.info['Start'] <=item[0]) & (self.info['End']>item[0]) ]
                    start_idx = row['Start'].values[0]
                    end_idx =row['End'].values[0]
                    out= perturb_local_noise(out, self.data, start_idx, end_idx)
                    #print(out.shape)
            else: 
                out=x

        return (out.reshape(1, *in_shape),)

    
    def save(self, out_file):
        """ Write a Image masker to a file stream.
        """
        super().save(out_file)

        # Increment the verison number when the encoding changes!
        with Serializer(out_file, "shap.maskers.Image", version=0) as s:
            s.save("mask_value", self.input_mask_value)
            s.save("shape", self.input_shape)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """ Load a Image masker from a file stream.
        """
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.Image", min_version=0, max_version=0) as s:
            kwargs["mask_value"] = s.load("mask_value")
            kwargs["shape"] = s.load("shape")
        return kwargs

def unpack_shap_explanation_contents(shap_values):
    values = getattr(shap_values, "hierarchical_values", None)
    if values is None:
        values = shap_values.values
    clustering = getattr(shap_values, "clustering", None)

    return values, clustering

def timeseries(shap_values, original, path= None):
    l=original.reshape(-1).shape[0]

    sns.set(rc={'figure.figsize':(15,6)})

    ax011 = plt.subplot(111)
    ax012 = ax011.twinx()
    sal_01=shap_values.values.reshape(-1) #np.abs(original.reshape(-1)-np.array(pop)[0][0].reshape(-1)).reshape(1,-1)
    print(len(np.asarray(sal_01).reshape(original.shape[0], original.shape[1])))
    print(np.asarray(sal_01).reshape(-1,1).shape)
    print(len(original.reshape(-1)))
    sns.heatmap(np.asarray(sal_01).reshape(1,-1), fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
    sns.lineplot(x=range(l), y=original.reshape(-1),color='white', ax=ax012)

    if path == None: 
        plt.show()
    else: 
        plt.savefig(path,transparent=True)
    plt.close()