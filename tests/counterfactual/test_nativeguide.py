import pytest
from tslearn.datasets import UCR_UEA_datasets
import sklearn
import pickle
import numpy as np 
import torch 
import tensorflow as tf
from ClassificationModels.CNN_T import ResNetBaseline, UCRDataset
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF \
     import NativeGuideCF


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
def cf_ng_torch_explainer( request, cnn_gunPoint_torch):
    # TODO 
    from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF \
     import NativeGuideCF
    X, y, model = cnn_gunPoint_torch
    cf_explainer =NativeGuideCF(model,(X,y),backend='PYT',method= request.param,mode='feat')
    yield X, y, model, cf_explainer

@pytest.fixture
def cf_ng_tensorflow_explainer( request, cnn_gunPoint_tensorflow):
    X, y, model = cnn_gunPoint_tensorflow
    cf_explainer =NativeGuideCF(model,(X,y),backend='TF',method= request.param,mode='time')
    yield X, y, model, cf_explainer



@pytest.mark.parametrize('cf_ng_torch_explainer',['dtw_bary_center','NG'],ids='method={}'.format,
                         indirect=True)
def test_cf_ng_torch_explainer(cf_ng_torch_explainer):

    # TODO ,'NUN_CF'
    X, y, model, cf = cf_ng_torch_explainer
    x = X[0].reshape(1,X.shape[-2], -1)

    probas = torch.nn.functional.softmax(model(torch.from_numpy(x).float())).detach().numpy()
    pred_class = probas.argmax()

    exp,ta = cf.explain(x, np.argmax(probas,axis=1))
    if exp is not None:
        assert exp.shape == (1,X.shape[-2], X.shape[-1])
        item=exp.reshape(1,X.shape[-2],-1)
        _item=  torch.from_numpy(item).float()
        model.eval()
        y_pred = torch.nn.functional.softmax(model(_item)).detach().numpy()
        y_label= np.argmax(y_pred)

        # check if target_class condition is met
        if ta is None:
            assert pred_class != y_label
        elif isinstance(ta, int):
            assert y_label == ta

@pytest.mark.parametrize('cf_ng_tensorflow_explainer',['NUN_CF','dtw_bary_center','NG'],ids='method={}'.format,
                         indirect=True)
def test_cf_ng_tensorflow_explainer(cf_ng_tensorflow_explainer):
    # TODO 
    X, y, model, cf = cf_ng_tensorflow_explainer
    x = X[0].reshape(1,X.shape[1],X.shape[2])
    probas = model.predict(x)
    pred_class = np.argmax(probas,axis=1)[0]

    exp,ta = cf.explain(x,np.argmax(probas,axis=1) )
    if exp is not None:
        assert exp.reshape(1,1,96).shape == (1,X.shape[-1], X.shape[-2])
        item=exp.reshape(1,X.shape[-2],-1)
        y_pred = model.predict(item)
        y_label= np.argmax(y_pred,axis=1)[0]

        # check if target_class condition is met
        if ta is None:
            assert pred_class != y_label
        elif isinstance(ta, int):
            assert y_label == ta