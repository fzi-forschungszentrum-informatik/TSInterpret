#From : https://www.kaggle.com/cuge1995/lstm-for-household-electric-power-cb225cfe-1
#Still TODO
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout


def model(train_x,train_y, test_x=None,test_y= None, neurons=[128], dropout=0):
    model = Sequential()
    for a in neurons:
        model.add(LSTM(a, input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_x, train_y, epochs=20, batch_size=70, validation_data=(test_x, test_y), verbose=2, shuffle=False)
    return model