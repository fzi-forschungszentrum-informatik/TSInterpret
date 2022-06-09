import sklearn.utils.validation
from sklearn.tree import DecisionTreeClassifier
from pyts.classification import TimeSeriesForest
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
import numpy as np
from sktime.classification.compose import ComposableTimeSeriesForestClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
# with sktime, we can write this as a pipeline
from sktime.transformations.panel.reduce import Tabularizer
from sktime.utils.slope_and_trend import _slope
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn import tree
#FROM https://www.sktime.org/en/stable/examples/02_classification_univariate.html
#Can return Feture Importance

#steps = [
    #(
    #    "extract",
    #    RandomIntervalFeatureExtractor(
    #        n_intervals="sqrt", features=[np.mean, np.std, _slope]
    #    ),
    #),
   # ("clf", TimeSeriesForestClassifier(n_estimators=100))#DecisionTreeClassifier()),
#]
#time_series_tree = Pipeline(steps)

def fitTimeSerieForest(X_train, y_train,X_test, y_test,save=None):
    '''Time Series Random Forest from pyts'''
    classifier = TimeSeriesForest(random_state=43)
    classifier.fit(X_train, y_train)
    classifier.score(X_test, y_test)
    time_series_tree = classifier
    print('Score ', time_series_tree.score(X_test, y_test))
    if save == None:
        return time_series_tree
    else:
        joblib.dump(time_series_tree, f'./ClassificationModels/models/{save}/TimeSeriesForest.pkl')
    return time_series_tree


def fitDecisionTree(X_train, y_train,X_test, y_test,save=None):
    '''Usual Decission Tree --> Careful nothing time-series specific'''
    classifier = DecisionTreeClassifier()#make_pipeline(Tabularizer(), RandomForestClassifier())
    classifier.fit(X_train, y_train)
    classifier.score(X_test, y_test)

    #time_series_tree = TimeSeriesForestClassifier(n_estimators=100)#Pipeline(steps)
    #time_series_tree.fit(X_train, y_train)
    print(sklearn.utils.validation.check_is_fitted(classifier))

    #tsf= ComposableTimeSeriesForestClassifier(
    #    estimator=time_series_tree,
    #    n_estimators=100,
    #    criterion="entropy",
    #    bootstrap=True,
    #    oob_score=True,
    #    random_state=1,
    #    n_jobs=-1,
    #)
    #tsf.fit(X_train, y_train)
    time_series_tree=classifier
    print('Score ',time_series_tree.score(X_test, y_test))
    if save== None:
        return time_series_tree
    else:
        joblib.dump(time_series_tree,f'./ClassificationModels/models/{save}/DecisionTree.pkl')
    return time_series_tree

def plot_feature_importance(model, save = None):
    '''Plots Feature Importance'''
    fi = model.feature_importances_
    # renaming _slope to slope.
    fi.rename(columns={"_slope": "slope"}, inplace=True)
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    fi.plot(ax=ax)
    ax.set(xlabel="Time", ylabel="Feature importance")
    plt.savefig(f'./Results/{save}/DecisionTreeFeatures.png')
    plt.close()

def plot_tree(model, save=None):
    '''Plot the decision Tree'''
    tree.plot_tree(model)
    plt.savefig(f'./Results/{save}/DecisionTreeStructure.png')
    plt.close()