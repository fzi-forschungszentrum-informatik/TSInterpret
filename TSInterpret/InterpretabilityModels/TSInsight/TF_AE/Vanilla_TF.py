from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tf_explain.core.integrated_gradients import IntegratedGradients
import tensorflow as tf 
class Vanilla_Autoencoder(Model):
    def __init__(self, input_size, latent_dim=128,architecture_fully=None):
        super(Vanilla_Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        if architecture_fully== None: 
            self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_size),
            layers.Reshape((-1,input_size[0]*input_size[1])),
            layers.Dense(input_size[0]*input_size[1], activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01)),
            layers.Dense(512, activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01)),
            layers.Dense(latent_dim, activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01))])

            self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01)),
            #layers.Dense(32, activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01)),
            layers.Dense(input_size[0]*input_size[1], activation="tanh",activity_regularizer=tf.keras.regularizers.L2(0.01)),
            layers.Dense(input_size[0]*input_size[1], activation="linear"), 
            layers.Reshape(input_size)])
        else: 
            #TODO This needs testing 
            architecture_encoder= [
            layers.InputLayer(input_shape=input_size),
            layers.Reshape((-1,input_size[0]*input_size[1])),
            layers.Dense(input_size[0]*input_size[1], activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01))]
            architecture_encoder.extend([layers.Dense(x, activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01))for x in architecture_fully])
            architecture_encoder.extend([layers.Dense(latent_dim, activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01))])

            architecture_decoder=[layers.Dense(x, activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.01))for x in architecture_fully.reverse()]
            architecture_decoder.extend([  layers.Dense(input_size[0]*input_size[1], activation="tanh",activity_regularizer=tf.keras.regularizers.L2(0.01)),
            layers.Dense(input_size[0]*input_size[1], activation="linear"), 
            layers.Reshape(input_size)])
            self.encoder = tf.keras.Sequential(architecture_encoder)
            self.decoder = tf.keras.Sequential(architecture_decoder)


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
