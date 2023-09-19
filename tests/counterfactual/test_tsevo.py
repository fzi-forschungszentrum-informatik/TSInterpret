import pytest
from tslearn.datasets import UCR_UEA_datasets
import numpy as np 
import torch
from ClassificationModels.CNN_T import ResNetBaseline,fit
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
import sklearn
import tensorflow as tf 


@pytest.fixture
def cnn_gunPoint_torch():
    X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset('ElectricDevices')
    sh=X_train.shape
    X_train=np.swapaxes(X_train,1,2)
    train_y=y_train
    test_y=y_test
    enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1))
    train_y=enc1.transform(train_y.reshape(-1,1))
    test_y=enc1.transform(test_y.reshape(-1,1))
    #train_y=enc1.transform(train_y.reshape(-1,1))
    model = ResNetBaseline(in_channels= X_train.shape[-2], num_pred_classes=len(np.unique(y_train)))
    model.load_state_dict(torch.load(f'./ClassificationModels/models/ElectricDevices/ResNet'))
    model.eval()
    return X_train, train_y, model

@pytest.fixture
def cnn_gunPoint_tensorflow():
    X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset('Coffee')
    sh=X_train.shape
    train_y=y_train
    test_y=y_test
    enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1))
    train_y=enc1.transform(train_y.reshape(-1,1))
    test_y=enc1.transform(test_y.reshape(-1,1))
    model = tf.keras.models.load_model(f'./ClassificationModels/models/Coffee/cnn/best_model.hdf5')
    return X_train, train_y, model

#@pytest.fixture
#def cnn_gunPoint_sklearn():
#    X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset('BasicMotions')
#    sh=X_train.shape
#    train_y=y_train
#    test_y=y_test
#    train_x=X_train
#    test_x=X_test
#    train_x = TimeSeriesScalerMinMax().fit_transform(train_x)
#    test_x = TimeSeriesScalerMinMax().fit_transform(test_x)
#    model = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
#    model.fit(train_x, train_y)
#    return train_x, train_y, model


@pytest.fixture
def cf_evo_torch_explainer( request, cnn_gunPoint_torch):
    X, y, model = cnn_gunPoint_torch
    cf_explainer =TSEvo(model= model,data=(X,y), backend='PYT',transformer=request.param, mode='feat',epochs=10)
    yield X, y, model, cf_explainer

@pytest.fixture
def cf_evo_tensorflow_explainer( request, cnn_gunPoint_tensorflow):
    #TO
    X, y, model = cnn_gunPoint_tensorflow
    cf_explainer =TSEvo(model= model,data=(X,y), backend='tf',epochs=10)
    yield X, y, model, cf_explainer
#@pytest.fixture
#def cf_ates_sklearn_explainer( request, cnn_gunPoint_sklearn):
#    X, y, model = cnn_gunPoint_sklearn
#    cf_explainer =AtesCF(model,(X,y),backend='SK',method= request.param,mode='time')
#    yield X, y, model, cf_explainer

@pytest.mark.parametrize('cf_evo_torch_explainer',['authentic_opposing_information','mutate_both','mutate_mean','frequency_band_mapping'],ids='transformer={}'.format,indirect=True)
def test_cf_evo_torch_explainer(cf_evo_torch_explainer):

    X, y, model, cf = cf_evo_torch_explainer
    x = X[0].reshape(1,X.shape[-2], -1)

    probas = torch.nn.functional.softmax(model(torch.from_numpy(x).float())).detach().numpy()
    pred_class = probas.argmax()

    exp,_ = cf.explain(x.reshape(1,1,-1),np.array([pred_class]))
    exp=np.array(exp)
    assert exp.reshape((1,X.shape[-2], X.shape[-1])).shape == (1,X.shape[-2], X.shape[-1])
    item=exp.reshape(1,X.shape[-2],-1)
    _item=  torch.from_numpy(item).float()
    model.eval()
    y_pred = torch.nn.functional.softmax(model(_item)).detach().numpy()
    y_label= np.argmax(y_pred)

   
    assert pred_class != y_label

#TODO ADD OPT
#@pytest.mark.parametrize('cf_evo_tensorflow_explainer',['brute'],ids='method={}'.format,
#                         indirect=True)
#@pytest.mark.parametrize('target',[None])#0,1
#def test_cf_evo_tensorflow_explainer(cf_evo_tensorflow_explainer,target):
 
#    X, y, model, cf = cf_evo_tensorflow_explainer
#    x = X[0].reshape(1,X.shape[1],X.shape[2])
#    probas = model.predict(x)
#    pred_class = np.argmax(probas,axis=1)[0]

#   exp,ta = cf.explain(x, target=target)
#    assert exp.shape == (1,X.shape[-1], X.shape[-2])
#    item=exp.reshape(1,X.shape[-2],-1)
#    y_pred = model.predict(item)
#    y_label= np.argmax(y_pred,axis=1)[0]

    # check if target_class condition is met
#    if ta is None:
#        assert pred_class != y_label
#    elif isinstance(ta, int):
#        assert y_label == ta
