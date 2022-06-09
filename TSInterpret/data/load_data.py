import numpy as np
import pandas as pd
import os
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from tslearn.datasets import UCR_UEA_datasets
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
# import tensorflow as tf
# import math
# import pickle
# import matplotlib.pylab as plt

# mpl.style.use('seaborn-paper')


### UTILITY CLASS FOR SEQUENCES SCALING ###

class Scaler1D:

    def fit(self, X):
        self.mean = np.nanmean(np.asarray(X).ravel())
        self.std = np.nanstd(np.asarray(X).ravel())
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return (X * self.std) + self.mean


### UTILITY CLASS FOR SEQUENCES SCALING ###

def min_max_scale(X, range=(-0.8, 0.8)):
    mi, ma = range
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (ma - mi) + mi
    return X_scaled


'''Change Train Data Split into balances 80 / 20 split, Further scaling options '''


# todo for clustering start categorical data at 0, Flexible cwd
def load_basic_dataset(dataset, window=-1, sep='\t', scaling='minmax', range=(-0.8, 0.8), cwd='../UCRArchive_2018/',mode='time'):
    '''
        Method to load Dataset from UCR
        Args:
            dataset (str): Name of Dataset .
            window (int): Time Window.
            sep(str): File Seperator Sring.
            scaling (str): Type of scaling (Currently only MinMax Support)-
            range(str): Range of scaler.
            cwd(str): Path to UCE Archieve
            mode(str): If first dimension are time steps : 'time', else 'feat'
        '''
    # cwd= os.getcwd()
    # print(cwd)
    #cwd = '../UCRArchive_2018/'
    # eliminated / data  after cwd/media/jacqueline/Data/UCRArchive_2018
    train = pd.read_csv(cwd + dataset + '/' + dataset + '_TRAIN.tsv', sep=sep, header=None)
    test = pd.read_csv(cwd + dataset + '/' + dataset + '_TEST.tsv', sep=sep, header=None)
    x_train = train.drop(columns=[0])
    y_train = train[0]
    x_test = test.drop(columns=[0])
    y_test = test[0]
    if scaling == 'minmax':
        x_train = min_max_scale(np.array(x_train), range)
        x_test = min_max_scale(np.array(x_test), range)
    #x_train = np.vstack((x_train, x_test))
    #y_train = np.concatenate((y_train, y_test))
    #x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    x_train= np.array(x_train)
    x_test=np.array(x_test)
    if mode =='time':
        x_train = x_train.reshape(-1, x_train.shape[-1], 1)
        x_test = x_test.reshape(-1,  x_test.shape[-1], 1)
    elif mode =='feat':
        x_train = x_train.reshape(-1,1,  x_train.shape[-1])
        x_test = x_test.reshape(-1,1,  x_test.shape[-1])
    return x_train, x_test, np.array(y_train), np.array(y_test)

def load_data_from_Kaggle(path='./bjoernjostein/ptb-diagnostic-ecg-database'):
    '''
    Method to load Dataset at
    Args:
        path (str): Path to File .

    Only works if API Token at
    TODO
    Not tested or finished
    '''
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(path)
    zf = ZipFile('titanic.zip')
    zf.extractall('./kaggle/') #save files in selected folder
    zf.close()
    return 'None'

def load_UEA_dataset(dataset, mode='time'):
    '''
        Method to load Dataset from UEA by using tslearn
        Args:
            dataset (str): Name of Dataset .
            mode(str): If first dimension are time steps : 'time', else 'feat'
        '''
    X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset(dataset)
    if mode =='time':
        train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
        test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    elif mode =='feat':
        train_x=X_train.reshape(-1,X_train.shape[-2],X_train.shape[-1])
        test_x=X_test.reshape(-1,X_train.shape[-2],X_train.shape[-1])
    return train_x,y_train,test_x,y_test

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''untested'''
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_multivariate_data(dataset='./data/Multivariate/household_power_consumption.txt', fillNaN='mean', scaler = 'MinMax', reframe= True,input_window=168, prediction_window = 12, granularity='hour'):
    'untested'
    df = pd.read_csv(dataset, sep=';', parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True,
                     low_memory=False, na_values=['nan', '?'], index_col='dt')
    org=df
    if fillNaN == 'mean':
        for j in range(0, 7):
            df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())
    df = df.resample('h').mean()
    if scaler=='MinMax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(df)
    if reframe == True:
        reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13]], axis=1, inplace=True)

    values = reframed.values
    n_train_time = 365 * 24
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    return  train_x, test_x, train_y, test_y, org


if __name__ == '__main__':
    #load_basic_dataset('ECG5000', 140, scaling='no')
    #load_multivariate_data('./Multivariate/household_power_consumption.txt')
    load_data_from_Kaggle()