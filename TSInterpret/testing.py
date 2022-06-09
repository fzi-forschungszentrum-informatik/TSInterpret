from data.load_data import load_basic_dataset
from InterpretabilityModels.lime.lime_timeseries import LimeTimeSeriesExplainer
from InterpretabilityModels.Cam.Cam import Cam
from InterpretabilityModels.GradCam.GradCam import GradCam
import numpy as np
import joblib
import  sklearn
import tensorflow as tf

def fit_classifier(classifier, x_train,y_train, x_test,y_test):


    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    classifier.fit(x_train, y_train, x_test, y_test, y_true)

def load_model(path):
    if path.endswith('.pkl'):
        model=joblib.load(path)
    else:
        model=tf.keras.models.load_model(path)
    return model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #Electric Devices , 96
    '''***Parameters***'''
    name = 'Coffee'
    window = 286
    num_features = 2 #Class Number ?
    num_slices = 17
    '''****************'''

    trainx, testx, trainy, testy=load_basic_dataset(name,window, scaling='None')

    '''FITS for different Models'''
    '''KNN'''
    #fitKnnModel(trainx.reshape(-1,window,1),trainy.reshape(-1),testx.reshape(-1,window,1),testy.reshape(-1),save=name)
    '''CNN Classification Model '''
    #model = Classifier_CNN(f'./ClassificationModels/models/{name}/', (trainx.shape[1], trainx.shape[2]), 2, True)
    #fit_classifier(model, trainx,trainy,testx,testy)
    '''Decision Tree'''
    #model=fitTimeSerieForest(trainx.reshape(-1,window),trainy.reshape(-1),testx.reshape(-1,window),testy.reshape(-1),save=name)
    #tree=fitDecisionTree(trainx.reshape(-1,window),trainy.reshape(-1),testx.reshape(-1,window),testy.reshape(-1),save=name)
    #tree=joblib.load('./ClassificationModels/models/Coffee/DecisionTree.pkl')
    #plot_tree(tree, save=name)
    #TODO does not work yet
    #plot_feature_importance(tree, save=name)
    '''If model exists Load the Model'''
    #tf.compat.v1.disable_eager_execution()
    model=load_model(f'./ClassificationModels/models/{name}/cnn/best_model.hdf5')
    '''Explainer Section'''
    #TODO Save Explainer and coresspondig Config
    #TODO with knew KNN Lime eXPLAINER NOT wORKING ANYMORE
    #TODO Lime Tabular Explainer ?
    '''Lime Test'''
    #explainer = LimeTimeSeriesExplainer(ml_model=model,num_features=num_features,num_slices=num_slices)  # class_name
    #predict_proba
    #exp = explainer.explain(testx[0].reshape(window,1), model.predict,replacement_method='noise')
    #explainer.plot_on_Sample(testx[0].reshape(-1),exp,(trainx,trainy), exp)
    #print(testx[0].reshape(window,1))
    #print(testx[0].reshape(window,1).shape)
    #metrics.roboustness(testx,explainer.explain_instance,model.predict)
    #metrics.faithfulness(trainx, trainy, testx,testy, explain, max_number_features=2)
    #exp = explainer.explain_instance(testx[0].reshape(window,1), model.predict, num_features=num_features,
    #                                 num_slices=num_slices, replacement_method='noise')
    #print(exp)
    #exp.as_pyplot_figure()
    #plt.show()
    #plt.close()
   
    '''Cam Test'''

    #explainer = Cam(mlmodel=model,mode='tensorflow')  # class_name
    #predict_proba
   # print(testx.shape)
    #exp,softmax = explainer.explain(testx)
    #explainer.plot_on_sample(exp,testx,testy,softmax,save=name)
    
    '''GradCam Test'''
    #tf.config.run_functions_eagerly(True)

    explainer = GradCam(mlmodel=model,mode='tensorflow')  # class_name
    #predict_proba
    print(testx.shape)
    exp = explainer.explain((testx,testy),1)
    explainer.plot_on_sample(exp)
