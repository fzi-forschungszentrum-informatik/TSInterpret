import numpy as np
from tslearn.datasets import UCR_UEA_datasets

data = UCR_UEA_datasets().load_dataset("ECG200")

import tensorflow.keras as keras

loaded_model = keras.models.load_model("./best_model.hdf5")

import tensorflow.keras as keras
from tslearn.datasets import UCR_UEA_datasets

loaded_model = keras.models.load_model("./best_model.hdf5")

# setscf = SETSCF(loaded_model, data, "TF", "time", time_contract_in_mins_per_dim=1)
from TSInterpret.InterpretabilityModels.counterfactual.SETS.class_shapelets import (
    get_class_shapelets,
)
from TSInterpret.InterpretabilityModels.counterfactual.SETS.sets import sets_explain
from TSInterpret.InterpretabilityModels.counterfactual.SETSCF import SETSCF

shapelets_distances = np.load(
    "TSInterpret/InterpretabilityModels/counterfactual/SETS/shapelets_distances.pkl",
    allow_pickle=True,
)
shapelets_distances_test = np.load(
    "TSInterpret/InterpretabilityModels/counterfactual/SETS/shapelets_distances_test.pkl",
    allow_pickle=True,
)
st_shapelets = np.load(
    "TSInterpret/InterpretabilityModels/counterfactual/SETS/shapelets.pkl",
    allow_pickle=True,
)

scores = np.load("TSInterpret/InterpretabilityModels/counterfactual/SETS/scores.npy")

train_x, train_y, test_x, test_y = data

train_x = np.swapaxes(train_x, 2, 1)
test_x = np.swapaxes(test_x, 2, 1)

data = train_x, train_y, test_x, test_y

(
    all_heat_maps,
    all_shapelets_class,
    all_shapelet_locations,
    all_shapelet_locations_test,
) = get_class_shapelets(
    data,
    data[0].shape[2],
    st_shapelets,
    shapelets_distances,
    shapelets_distances_test,
)

ts_instance = 4

exp, label = sets_explain(
    ts_instance,
    loaded_model,
    data,
    data[0].shape[2],
    st_shapelets,
    all_shapelet_locations,
    all_shapelet_locations_test,
    all_shapelets_class,
    all_heat_maps,
    scores,
)
