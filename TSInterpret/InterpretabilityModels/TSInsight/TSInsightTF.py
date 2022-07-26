from telnetlib import RCP
from zmq import device
from TSInterpret.InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
from TSInterpret.InterpretabilityModels.TSInsight.TSInsight import TSInsight
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tf_explain.core.integrated_gradients import IntegratedGradients
import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from tf_explain.core.gradients_inputs import GradientsInputs
from tensorflow.python.ops.numpy_ops import np_config
#from alibi.explainers import IntegratedGradients
np_config.enable_numpy_behavior()
#TODO ALibi, Deep explain , Skater
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

class TSInsightTF(TSInsight):
    '''
        Instantiated TS Insight
        mlmodel: Machine Learning Model to be explained.
        mode : Second dimension is feature --> 'feat', is time --> 'time'
        backend: PYT or TF
        autoencode: None or instance of an implemented and traine AE 
        data: In case of TF a Tuple, in case of PYT a DataLoader 
    '''
    def __init__(self, mlmodel,shape,data, test_data=None, mode='time', backend='TF',autoencoder = None, device = 'cpu',**kwargs):
        super().__init__(mlmodel,shape, mode, backend)
        self.device=device
        self.model.trainable = False
        self.saliency=IntegratedGradients()

        if autoencoder == None:
            self.autoencoder=Vanilla_Autoencoder(shape)
            self._train(data,test_data,**kwargs)
            self.fine_step(data,**kwargs)
        else: 
            self.autoencoder=autoencoder
            self._fine_tuning(data,**kwargs)
        


    def explain(self, item, flatten=True):
        if flatten:
            item = item.reshape(-1,self.shape[0],self.shape[1])
        output= self.autoencoder.predict(item)
        return output
    
    #def l1_penalty(self,batch):
    #    layers= list(self.autoencoder.children())
    #    loss = 0
    #    values = batch.float()
    #    for i in range(len(layers)):
    #        values = F.relu((layers[i](values)))
    #        loss += torch.mean(torch.abs(values))
    #    return loss


    def _train(self, dataloader, test_dataloader=None, epochs=1000, flatten=True, loss_fn='mse', lr=0.01, reduction_factor=0.9, reduction_tolerance=4, patience=10, lam = 0.2,l2 = True, batch_size=32):
        #TODO restore best weights
        #TODO L1 = False is correct

        x_train,_=dataloader
        print('here',x_train.shape)
        x_test,_=test_dataloader
        callback=tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=patience,restore_best_weights=True)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000,  decay_rate=reduction_factor)


        opt=tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.autoencoder.compile(optimizer=opt, loss=loss_fn)
        self.autoencoder.fit(x_train,x_train, epochs=epochs,batch_size=batch_size, shuffle=True,  validation_data=(x_test, x_test),callbacks=[callback])



    def _hyperparameter(self,batch,target):
        '''
        #TODO check Calc
        #TODO More Efficiency?
    #   '''
        gamma=[]
        ßeta=[]
        #for x,y in zip(batch, target):
        #    print(x.shape)
        #    print(x)
        x=batch 
        y=target
        #print(x.shape)
        x=x.reshape(-1,self.shape[0],self.shape[1],1)
            #print(x.shape)
        sal= self.saliency.explain((x,None),self.model,class_index=np.argmax(y))
        #sal=sal[0].detach().numpy()
        I= (sal- np.ones_like(sal)*np.min(sal))/( np.ones_like(sal)*np.max(sal)- np.ones_like(sal)*np.min(sal))
        gamma.append(I)
        ßeta.append(np.ones_like(I)-I)
        return np.array(ßeta),np.array(gamma)
    

    
    def _fine_tuning(self, dataloader, epochs=50, flatten=True, lr=0.0001, reduction_factor=0.9, reduction_tolerance=4, patience=10,l1=True,ß=0.0001,om=4.0,lam=0.2, C=10, self_tune = True, batch_size=32):
        pass
        #loss_fn1=tf.keras.losses.CategoricalCrossentropy()

        #loss_fn2='mse'
        
        #x_train,_=dataloader
        #print('here',x_train.shape)
        #callback=tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=patience,restore_best_weights=True)

        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000,  decay_rate=reduction_factor)


        #opt=tf.keras.optimizers.Adam(learning_rate=lr)

        #loss_fn=self._fine_tuning_loss()        
        #self.autoencoder.compile(optimizer=opt, loss=loss_fn)
        #self.autoencoder.fit(x_train,x_train, epochs=epochs,batch_size=batch_size, shuffle=True,  validation_data=(x_test, x_test),callbacks=[callback])
    
    #def _hyperparameter(self,batch,target):#

    #    gamma=[]
     #   ßeta=[]
     #   for x,y in zip(batch, target):
            #x=x.reshape(-1,self.shape[0],self.shape[1]).float()
            #TODO Attribution Method
     #       print(np.argmax(y))
     #       #TODO weights from AE or Classificator ? --> Classifcator 1D - Conv --> Classical methods should not be an issue 
     #       print(self.model.layers[1].weights)
     #       sal= self.model.layers[1].weights[0][1]#self.saliency.explain((x,None),model = self.model,class_index= np.argmax(y))#.attribute(x,np.argmax(y))
     #       #sal=sal[0].detach().numpy()
      #      I= (sal- np.ones_like(sal)*np.min(sal))/( np.ones_like(sal)*np.max(sal)- np.ones_like(sal)*np.min(sal))
     #       gamma.append(I)
     #       ßeta.append(np.ones_like(I)-I)
     #   return np.array(ßeta),np.array(gamma)


    def fine_step(self, data,C=10,lr=0.0001, reduction_factor=0.9, reduction_tolerance=4, patience=10,epochs= 2):
        #TODO Batch enable
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000,  decay_rate=reduction_factor)


        opt=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        loss_fn1=tf.keras.losses.CategoricalCrossentropy()

        loss_fn2=tf.keras.losses.MeanSquaredError()
        data,labels = data
        train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x, labels) in enumerate(train_dataset):
                print('X',x.shape)
                x=x.reshape(-1,x.shape[0],x.shape[1])

                ßeta,gamma =self._hyperparameter(x,labels)

                with tf.GradientTape() as tape:
                    #TODO Calculation data vs x 
                    reconstruction = self.autoencoder(np.array(x).reshape(-1,self.shape[0],self.shape[1]), training=True)
                    reconstruction_loss = tf.reduce_mean(loss_fn2(reconstruction*gamma,x*gamma))
                    print('reconstruction_loss',reconstruction_loss)
                    classification_loss = loss_fn1(self.model(reconstruction).reshape(-1),labels)
                    print('classification_loss',classification_loss)
                    total_loss = classification_loss+reconstruction_loss+ C*np.sum(np.abs(self.autoencoder(x,training=True)*ßeta))
                    print('totel',total_loss)
                grads = tape.gradient(total_loss, self.autoencoder.trainable_weights)
                print('Grads',grads)

                opt.apply_gradients(zip(grads, self.autoencoder.trainable_weights))


