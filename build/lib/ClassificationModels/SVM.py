from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.utils import to_time_series_dataset
from tslearn.svm import TimeSeriesSVC
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.preprocessing import OneHotEncoder
'''
Stuff taken from Leftis Code
TODO 
    * Still needs rework testes. 
    * Which Items are necessary 
    * Which are duplicate 
'''

def learning_shapelet (x_train,y_train):
    """
    Create and fit learning shapelet classifier. Method from [1] tslearn

    Parameters:
        x_train (np.ndarray): time series for the training.
        y_train (np.ndarray): label of the time series for the training.

    Returns:
        fitted Learning Shapelet classifier

    """

    # ensure to get the data as the right format
    if len(x_train.shape) < 3:
        x_train = to_time_series_dataset(x_train)

    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=x_train.shape[0],
                                                           ts_sz=x_train.shape[1],
                                                           n_classes=len(set(y_train)),
                                                           l=0.1,
                                                           r=10)

    ts_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                            optimizer=Adagrad(lr=.1),
                            weight_regularizer=.01,
                            max_iter=50,
                            verbose_level=0)

    ts_clf.fit(x_train, y_train)
    return ts_clf

def timeseriessvc(x_train,y_train):
    """
    Create and fit svm classifier. Method from [1] tslearn

    Parameters:
        x_train (np.ndarray): time series for the training.
        y_train (np.ndarray): label of the time series for the training.

    Returns:
        fitted SVM classifier

    """
    #sz=len(x_train[0]) does not exist 
    ts_clf = TimeSeriesSVC(degree=1, kernel="rbf", probability=True)
    ts_clf.fit(x_train, y_train)
    return ts_clf

def dtw_wdistance_1NN(x_train,y_train):
    """
    Create and fit 1NN DTW classifier weighted by the distance. Method from [1] tslearn

    Parameters:
        x_train (np.ndarray): time series for the training.
        y_train (np.ndarray): label of the time series for the training.

    Returns:
        fitted 1NN DTW classifier

    """
    ts_clf = KNeighborsTimeSeriesClassifier(metric="dtw",n_neighbors=1,weights='distance')
    ts_clf.fit(x_train, y_train)
    return ts_clf

def fastdtw_wdistance_1NN(x_train,y_train):
    """
    Create and fit 1NN fast DTW classifier weighted by the distance. Method from [1] tslearn

    Parameters:
        x_train (np.ndarray): time series for the training.
        y_train (np.ndarray): label of the time series for the training.

    Returns:
        fitted 1NN DTW classifier

    """
    ts_clf = KNeighborsTimeSeriesClassifier(metric="fastdtw",n_neighbors=1,weights='distance')
    ts_clf.fit(x_train, y_train)
    return ts_clf

def euclidean_wdistance_1NN(x_train,y_train):
    """
    Create and fit 1NN euclidean distance classifier weighted by the distance. Method from [1] tslearn

    Parameters:
        x_train (np.ndarray): time series for the training.
        y_train (np.ndarray): label of the time series for the training.

    Returns:
        fitted 1NN euclidean distance classifier

    """
    ts_clf = KNeighborsTimeSeriesClassifier(metric="euclidean",n_neighbors=1,weights='distance')
    ts_clf.fit(x_train, y_train)
    return ts_clf