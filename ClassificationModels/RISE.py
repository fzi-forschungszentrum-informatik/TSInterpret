from sktime.classification.interval_based import RandomIntervalSpectralForest
import joblib

def fit_rise(X_train, y_train,X_test, y_test,save = None ):
    rise = RandomIntervalSpectralForest(n_estimators=10)
    rise.fit(X_train, y_train)
    rise.score(X_test, y_test)
    if save== None:
        return rise
    else:
        joblib.dump(f'./models/{save}/Rise.pkl')