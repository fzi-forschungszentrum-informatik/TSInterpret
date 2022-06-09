from InterpretabilityModels.InterpretabilityBase import InterpretabilityMethod
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage.interpolation import zoom
'''
https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
PyTorch: https://git.pandolar.top/jacobgil/pytorch-grad-cam

TODO 
* Check Implementation if it is really Grad Cam 
* Reuse Activation of gradient_importance and activation_grad
* https://github.com/cordeirojoao/ECG_Processing/blob/master/Ecg_keras_v9-Raphael.ipynb , https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54
* from tf_explain.core.grad_cam import GradCAM https://gilberttanner.com/blog/interpreting-tensorflow-model-with-tf-explain

'''

#def gradient_importance(seq, model):   
#    '''Taken From https://towardsdatascience.com/feature-importance-with-time-series-and-recurrent-neural-network-27346d500b9c''' 
#    seq = tf.Variable(seq[np.newaxis,:,:], dtype=tf.float32)    
#    with tf.GradientTape() as tape:
#        predictions = model(seq)    
#    grads = tape.gradient(predictions, seq)
#    grads = tf.reduce_mean(grads, axis=1).numpy()[0]
#    
#    return grads

#def activation_grad(seq, model):
#    
#    seq = seq[np.newaxis,:,:]
#    # TODO no Layer named extractor 
#    grad_model = tf.keras.Model([model.inputs], 
#                       [model.get_layer('extractor').output, 
#                        model.output])    
    # Obtain the predicted value and the intermediate filters
#    with tf.GradientTape() as tape:
#        seq_outputs, predictions = grad_model(seq)    
    # Extract filters and gradients
#    output = seq_outputs[0]
#    grads = tape.gradient(predictions, seq_outputs)[0]    
    # Average gradients spatially
#    weights = tf.reduce_mean(grads, axis=0)    
    # Get a ponderated map of filters according to grad importance
#    cam = np.ones(output.shape[0], dtype=np.float32)
#    for index, w in enumerate(weights):
#        cam += w * output[:, index]    
#    time = int(seq.shape[1]/output.shape[0])
#    cam = zoom(cam.numpy(), time, order=1)
#    heatmap = (cam - cam.min())/(cam.max() - cam.min())
    
#    return heatmap

class Cam(InterpretabilityMethod):
    """
    Abstract class to implement custom interpretability methods
    ----------
    mlmodel:

    Methods
    -------
    explain:
        explains a instance
    Returns
    -------
    None

    TODO 
    -------
        * Only Tensorflow 
        * Is the Implementation Suitable ? 
        * Works for all Models or only FCN ? 
    Restrictions: 
    -------
    - Model needs to have a Gradient ! 

    Source:
    -------
    https://arxiv.org/abs/1512.04150

    """

    def __init__(self, mlmodel,mode):
        self.model_to_explain = mlmodel
        self.mode = mode 

    def explain(self, data):
        """
        Generate counterfactual examples for given factuals.
        Parameters
        ----------
        instance: np.array
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).
        Returns
        -------
        pd.DataFrame
            Encoded and normalised counterfactual examples.

        TODO 
            * why do I need Data here ? 

        """
        #TODO used to be -3 , depends on number of conv layers 
       
        get_last_conv = keras.backend.function([self.model_to_explain.layers[0].input, keras.backend.learning_phase()], [self.model_to_explain.layers[-2].output])
        last_conv = get_last_conv([data[:100], 1])[0]

        get_softmax = keras.backend.function([self.model_to_explain.layers[0].input, keras.backend.learning_phase()], [self.model_to_explain.layers[-1].output])
        softmax = get_softmax(([data[:100], 1]))[0]
        softmax_weight = self.model_to_explain.get_weights()[-2]
        print(softmax_weight.shape)
        print(last_conv.shape)
        #exp=softmax_weight
        exp = np.dot(last_conv, softmax_weight)    
        print(exp.shape)    
        return exp, softmax

    def plot_on_sample(self,exp,x_test,y_test,softmax, save=None):
        """
        Plots expalantion on the explained Sample. 

        Parameters
        ----------
        instance: np.array
            timeseries instance in two-dimensional shape (m, n).
        exp: expalantaion 

        Returns
        -------
        matplotlib.pyplot.Figure
            Visualization of Expalanation.

        """
        #TODO SET Range Automatically to number Features 
        time= int(x_test.shape[1]/exp.shape[1])#286
        for k in range(12):
            #print(k)
            CAM= zoom(exp, time, order=1)
            #print(exp.shape)
            #print(CAM.shape)
            #Normelize explanation
            CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
            #print(CAM.shape)
            c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
            #print(c)
            plt.figure(figsize=(13, 7))
            plt.plot(x_test[k].squeeze())
            #TODO eliminated two dimensions drom scatter
            plt.scatter(np.arange(len(x_test[k])), x_test[k].squeeze(), cmap='hot_r',c=c[k].squeeze(), s=100)# two : are missing # s = used to be 100 
            plt.title(
                'True label:' + str(y_test[k]) + '   likelihood of label ' + str(y_test[k]) + ': ' + str(softmax[k][int(y_test[k])]))

            plt.colorbar()

            if save != None:
                print(k)
                plt.savefig(f'./Results/{save}/Cam_{k}.png')
            else:
                plt.show()
