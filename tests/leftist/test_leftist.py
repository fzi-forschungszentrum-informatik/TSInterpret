import pytest
from tslearn.datasets import UCR_UEA_datasets
import sklearn
import pickle
import numpy as np 
import torch 
import tensorflow as tf
from ClassificationModels.CNN_T import ResNetBaseline, UCRDataset
from TSInterpret.InterpretabilityModels.leftist.leftist import LEFTIST

@pytest.fixture
def cnn_gunPoint_torch():
    X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset('GunPoint')
    sh=X_train.shape
    X_train=X_train.reshape(sh[0], sh[-1],-1)
    train_y=y_train
    test_y=y_test
    enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1))
    train_y=enc1.transform(train_y.reshape(-1,1))
    test_y=enc1.transform(test_y.reshape(-1,1))
    model = ResNetBaseline(in_channels= X_train.shape[-2], num_pred_classes=len(np.unique(y_train)))
    model.load_state_dict(torch.load(f'./ClassificationModels/models/GunPoint/ResNet'))
    model.eval()
    return X_train, train_y, model

@pytest.fixture
def cnn_gunPoint_tensorflow():
    X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset('ElectricDevices')
    sh=X_train.shape
    train_y=y_train
    test_y=y_test
    enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1))
    train_y=enc1.transform(train_y.reshape(-1,1))
    test_y=enc1.transform(test_y.reshape(-1,1))
    model = tf.keras.models.load_model(f'./ClassificationModels/models/ElectricDevices/cnn/best_model.hdf5')
    return X_train, train_y, model

@pytest.fixture
def leftist_torch_explainer( request, cnn_gunPoint_torch):
    X, y, model = cnn_gunPoint_torch
    shape=X[0].shape
    cf_explainer = LEFTIST(model,(X,None),mode='feat', backend='PYT', learning_process_name=request.param['a'],transform_name=request.param['b'])
    yield X, y, model, cf_explainer

@pytest.fixture
def leftist_tensorflow_explainer( request, cnn_gunPoint_tensorflow):
    X, y, model = cnn_gunPoint_tensorflow
    shape=X[0].shape
    cf_explainer = LEFTIST(model,(X,y),mode='time',backend='TF',transform_name=request.param['b'],learning_process_name=request.param['a'])
    yield X, y, model, cf_explainer


@pytest.mark.parametrize("leftist_torch_explainer", [{"a": "Lime", "b": "uniform"},{"a": "Lime", "b": "straight_line"},
    {"a": "Lime", "b": "background"}, {"a": "Shap", "b": "uniform"},{"a": "Shap", "b": "straight_line"},{"a": "Shap", "b": "background"}],
                         indirect=True)
def test_leftist_torch_explainer(leftist_torch_explainer):
    X, _, _, method = leftist_torch_explainer
    x = X[0].reshape(1,X.shape[-2], -1)
    exp= method.explain(x)
    assert np.array(exp).shape == (2, X.shape[-1])

@pytest.mark.parametrize("leftist_tensorflow_explainer", [{"a": "Lime", "b": "uniform"},{"a": "Lime", "b": "straight_line"},
    {"a": "Lime", "b": "background"}, {"a": "Shap", "b": "uniform"},{"a": "Shap", "b": "straight_line"},{"a": "Shap", "b": "background"}],
                         indirect=True)
def test_leftist_tensorflow_explainer(leftist_tensorflow_explainer):
    X, _ , _ , method = leftist_tensorflow_explainer
    x = X[0].reshape(1,X.shape[1],X.shape[2])
    exp =method.explain(x)
    assert np.array(exp).shape == (7, X.shape[-2])
