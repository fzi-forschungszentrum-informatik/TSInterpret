from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score as acc
import joblib
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

def fitKnnModel(trainx,trainy,testx=None,testy=None,save=None):
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")
    knn.fit(trainx, trainy)

    if save != None:
        joblib.dump(knn,f'./ClassificationModels/models/{save}/knn_Model.pkl')
    #if testx!= None:
    print(f'score {knn.score(testx, testy)}')
    print('Accuracy KNN for coffee dataset: %f' % (acc(testy, knn.predict(testx))))
    return knn
