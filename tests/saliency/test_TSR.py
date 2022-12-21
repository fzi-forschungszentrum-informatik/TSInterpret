import pytest
from tslearn.datasets import UCR_UEA_datasets
import sklearn
import pickle
import numpy as np 
import torch 
import tensorflow as tf
from ClassificationModels.CNN_T import ResNetBaseline, UCRDataset
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR

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
    X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset('BasicMotions')
    sh=X_train.shape
    train_y=y_train
    test_y=y_test
    enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1))
    train_y=enc1.transform(train_y.reshape(-1,1))
    test_y=enc1.transform(test_y.reshape(-1,1))
    model = tf.keras.models.load_model(f'./ClassificationModels/models/BasicMotions/cnn/BasicMotionsbest_model.hdf5')
    return X_train, train_y, model

@pytest.fixture
def tsr_torch_explainer( request, cnn_gunPoint_torch):
    X, y, model = cnn_gunPoint_torch
    shape=X[0].shape
    cf_explainer = TSR(model, X.shape[-1],X.shape[-2], method=request.param,  mode='feat')
    yield X, y, model, cf_explainer

@pytest.fixture
def tsr_tensorflow_explainer( request, cnn_gunPoint_tensorflow):
    X, y, model = cnn_gunPoint_tensorflow
    shape=X[0].shape
    cf_explainer = TSR(model, X.shape[-2],X.shape[-1], method=request.param,  mode='time')
    yield X, y, model, cf_explainer

#TODO 'IG','GS','DL','DLS','SG','SVS','FA','FO'
@pytest.mark.parametrize("tsr_torch_explainer", ['GRAD','IG','GS','DLS','SG'],
                         indirect=True)
@pytest.mark.parametrize('tsr',[True,False])
def test_tsr_torch_explainer(tsr_torch_explainer,tsr):
    X, y, _, method = tsr_torch_explainer
    x = np.array([X[0,:,:]])
    exp= method.explain(x,labels=int(np.argmax(y[0])),TSR = tsr)
    assert np.array(exp).shape == (1, X.shape[-1])

@pytest.mark.parametrize("tsr_tensorflow_explainer", ['GRAD','IG','GS','DLS','SG'],
                         indirect=True)
@pytest.mark.parametrize('tsr',[True,False])
def test_tsr_tensorflow_explainer(tsr_tensorflow_explainer,tsr):
    X, y , _ , method = tsr_tensorflow_explainer
    x = np.array([X[0,:,:]])
    exp= method.explain(x,labels=int(np.argmax(y[0])),TSR = tsr)
    assert np.array(exp).shape == (X.shape[-2], X.shape[-1])

