import numpy as np

__author__ = 'Mael Guilleme mael.guilleme[at]irisa.fr'
from TSInterpret.InterpretabilityModels.utils import tensorflow_wrapper, torch_wrapper,sklearn_wrapper

def predict_proba(neighbors, model_to_explain,backend, mode):
    """
    Classify the generated neighbors by the model to explain as in LIME.

    Parameters:
        neighbors (Neighbors): the neighbors.
        model_to_explain (..): the model to explain.

    Returns:
        neighbors (Neighbors): add the classification of the neighbors by the model to explain.
    """
    print('This is prediction')

    # classify neighbors by the model to explain
    if mode == 'time':
        neighbors.values=neighbors.values.reshape(-1,neighbors.values.shape[-2],neighbors.values.shape[-1] )
        items=neighbors.values.reshape(-1,neighbors.values.shape[-1],neighbors.values.shape[-2] )
    elif mode == 'feat':
        neighbors.values=neighbors.values.reshape(-1,neighbors.values.shape[-2],neighbors.values.shape[-1] )
        items=neighbors.values.reshape(-1,neighbors.values.shape[-1],neighbors.values.shape[-2] )

    if backend == 'SK' :
        neighbors.proba_labels = np.array(sklearn_wrapper(model_to_explain,mode).predict(items))
    elif backend == 'TF':
        neighbors.proba_labels = np.array(tensorflow_wrapper(model_to_explain,mode).predict(items))
    else: 
        neighbors.proba_labels = np.array(torch_wrapper(model_to_explain,mode).predict(items))
    print(neighbors.values.shape)
    #print(neighbors.values)
    #print(model_to_explain.predict_proba(neighbors.values))
    
    #print(neighbors.proba_label)
    if len(neighbors.proba_labels[0]) == 1:
        neighbors.proba_labels = np.array([np.array([el[0],1-el[0]]) for el in neighbors.proba_labels])
    return neighbors

def reconstruct(neighbors, transform):
    """
    Build the values of the neighbors in the original data space of the instance to explain.
    Store the values into neighbors value as a dictionary.

    Parameters:
        neighbors_masks (np.ndarray): masks of the neighbors.
        transform (Transform): the transform function.

    Returns:
        neighbors_values (np.ndarray): values of the neighbors in the original data space of the instance to explain.
    """
    print(neighbors)
    neighbors_values = np.apply_along_axis(transform.apply, 1, neighbors.masks)

    dict_neighbors_value = {}
    for idx in range(len(neighbors_values)):
        dict_neighbors_value[idx] = neighbors_values[idx]

    neighbors.values = dict_neighbors_value

    return neighbors_values


