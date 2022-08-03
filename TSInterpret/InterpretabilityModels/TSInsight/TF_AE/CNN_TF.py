from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tf_explain.core.integrated_gradients import IntegratedGradients
import tensorflow as tf 
class CNN_Autoencoder(Model):
    def __init__(self, input_size, latent_dim=128,architecture_fully=None):
        super(CNN_Autoencoder, self).__init__()
        self.latent_dim = latent_dim    
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_size),
            layers.Conv1D(8,3, activation='relu', padding='same',dilation_rate=2),
            layers.MaxPooling1D(2),
            layers.Conv1D(4,3, activation='relu', padding='same',dilation_rate=2),
            layers.MaxPooling1D(2),
            layers.AveragePooling1D(),
            layers.Flatten(),
            layers.Dense(2)
            ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(64),
            layers.Reshape((16,4)),
            #layers.Dense(32, activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01)),
            layers.Conv1D(4,1,strides=1, activation='relu', padding='same'),
            layers.UpSampling1D(2), 
            layers.Conv1D(8,1,strides=1, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.UpSampling1D(2),
            layers.Conv1D(1,1,strides=1, activation='linear', padding='same')
            
            ])


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
