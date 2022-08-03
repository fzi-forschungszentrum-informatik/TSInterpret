from TSInterpret.InterpretabilityModels.TSInsight.TSInsight import TSInsight
from TSInterpret.InterpretabilityModels.TSInsight.TF_AE.Vanilla_TF import Vanilla_Autoencoder
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
#from tensorflow.keras.callbacks import ReduceLROnPlateau
from TSInterpret.InterpretabilityModels.TSInsight.TF_AE.LROnPlateau import CustomReduceLROnPlateau as ReduceLROnPlateau
#from alibi.explainers import IntegratedGradients
np_config.enable_numpy_behavior()
#TODO ALibi, Deep explain , Skater

class TSInsightTF(TSInsight):
    '''
        Instantiated TS Insight
        mlmodel: Machine Learning Model to be explained.
        mode : Second dimension is feature --> 'feat', is time --> 'time'
        backend: PYT or TF
        autoencode: None or instance of an implemented and traine AE 
        data: In case of TF a Tuple, in case of PYT a DataLoader 
    '''
    def __init__(self,  mlmodel,shape,data, test_data=None, backend='TF',autoencoder = None, device = 'cpu',loss_fn='mse', lr=0.001,**kwargs):
        mode ='time'
        super().__init__(mlmodel,shape, mode, backend)
        self.device=device
        self.model.trainable = False
        self.saliency=IntegratedGradients()
        x_train,y_train=data
        self.shape=(x_train.shape[-2], x_train.shape[-1])

        if autoencoder is None:
            self.autoencoder=Vanilla_Autoencoder(shape)
            self._train(data,test_data,**kwargs)
        elif autoencoder =='cnn':
            #TODO
            pass
        elif autoencoder =='recurrent':
            #TODO
            pass
        else: 
            self.autoencoder=autoencoder
        self._fine_tuning(data,test_data,**kwargs)
        


    def explain(self, item, flatten=True):
        if flatten:
            item = item.reshape(-1,self.shape[0],self.shape[1])
        output= self.autoencoder.predict(item)
        return output


    def _train(self, dataloader, test_dataloader=None, epochs=1000, loss_fn='mse', lr=0.001, patience=20,weight_decay=1e-5,batch_size=1):
        #TODO restore best weights
        #TODO L1 = False is correct

        x_train,_=dataloader
        x_test,_=test_dataloader
        callback=tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=patience,restore_best_weights=True)
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000,  decay_rate=weight_decay)
        opt=tf.keras.optimizers.Adam(learning_rate=lr)

        self.autoencoder.compile(optimizer=opt, loss=loss_fn)
        self.autoencoder.fit(x_train,x_train, epochs=epochs,batch_size=batch_size, shuffle=True,  validation_data=(x_test, x_test),callbacks=[callback])



    def _hyperparameter(self,batch,target):
        '''
        #TODO check Calc
        '''
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
    

    
    #def _fine_tuning(self, dataloader, epochs=50, flatten=True, lr=0.0001, reduction_factor=0.9, reduction_tolerance=4, patience=10,l1=True,ß=0.0001,om=4.0,lam=0.2, C=10, self_tune = False, batch_size=32):
    #    loss_fn1=tf.keras.losses.CategoricalCrossentropy()#

     #   loss_fn2='mse'
     #   
     #   x_train,_=dataloader
     #   #TODO 
     #   callback=tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=patience,restore_best_weights=True)#

     #   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000,  decay_rate=reduction_factor)


        #opt=tf.keras.optimizers.Adam(learning_rate=lr)

        #loss_fn=self._fine_tuning_loss()        
        #self.autoencoder.compile(optimizer=opt, loss=loss_fn)
        #self.autoencoder.fit(x_train,x_train, epochs=epochs,batch_size=batch_size, shuffle=True,  validation_data=(x_test, x_test),callbacks=[callback])
    


    def _fine_tuning(self, dataloader,test_data, epochs=50, lr=0.0001, patience=10,l1=True,ß=0.0001,om=4.0,lam=0.2, C=10, self_tune = False):
        #TODO use Corect Scheduler
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000,  decay_rate=reduction_factor)
        opt=tf.keras.optimizers.Adam(learning_rate=lr)
        reduce_rl_plateau = ReduceLROnPlateau(patience=4,  
                              factor=0.9,
                              verbose=1, 
                              optim_lr=opt.learning_rate)

        
        loss_fn1=tf.keras.losses.CategoricalCrossentropy()

        loss_fn2=tf.keras.losses.MeanSquaredError()
        data,labels = dataloader
        train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        data, labels= test_data
        test_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        trigger_times=0
        best_model=self.autoencoder
        best_loss=1000

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            reduce_rl_plateau.on_train_begin()
            for step, (x, labels) in enumerate(train_dataset):
                #print('X',x.shape)
                x=x.reshape(-1,x.shape[0],x.shape[1])

                ßeta,gamma =self._hyperparameter(x,labels)



                with tf.GradientTape() as tape:
                    #TODO Calculation data vs x 
                    reconstruction = self.autoencoder(np.array(x).reshape(-1,self.shape[0],self.shape[1]), training=True)
                    l2_reg=tf.add_n([ tf.nn.l2_loss(v) for v in self.autoencoder.weights
                    if 'bias' not in v.name ])
                    
                    if self_tune:
                        
                        reconstruction_loss = tf.reduce_mean(loss_fn2(reconstruction*gamma,x*gamma))
                        #print('reconstruction_loss',reconstruction_loss)
                        classification_loss = loss_fn1(self.model(reconstruction).reshape(-1),labels)
                        #print('classification_loss',classification_loss)
                        total_loss = classification_loss+reconstruction_loss+ C*np.sum(np.abs(self.autoencoder(x,training=True)*ßeta))
                        #print('totel',total_loss)
                    else:
                        reconstruction_loss = tf.reduce_mean(loss_fn2(reconstruction,x))*om
                        #print('reconstruction_loss',reconstruction_loss)
                        classification_loss = loss_fn1(self.model(reconstruction).reshape(-1),labels)
                        #print('classification_loss',classification_loss)
                        loss3= ß* tf.reduce_sum(np.abs( reconstruction)) 
                        #l2= lam* l2_reg
                        l2=0
                        total_loss = classification_loss+reconstruction_loss+ loss3 +l2
                        #print('totel',total_loss)
                grads = tape.gradient(total_loss, self.autoencoder.trainable_weights)

                opt.apply_gradients(zip(grads, self.autoencoder.trainable_weights))

            #DO THE Testing 
            for step, (x, labels) in enumerate(test_dataset):
                #print('X',x.shape)
                x=x.reshape(-1,x.shape[0],x.shape[1])

                ßeta,gamma =self._hyperparameter(x,labels)


                #TODO Calculation data vs x 
                reconstruction = self.autoencoder(np.array(x).reshape(-1,self.shape[0],self.shape[1]), training=True)
                l2_reg=tf.add_n([ tf.nn.l2_loss(v) for v in self.autoencoder.weights
                    if 'bias' not in v.name ])
                    
                if self_tune:
                        
                    reconstruction_loss_val = tf.reduce_mean(loss_fn2(reconstruction*gamma,x*gamma))
                    #print('reconstruction_loss',reconstruction_loss)
                    classification_loss_val = loss_fn1(self.model(reconstruction).reshape(-1),labels)
                    #print('classification_loss',classification_loss)
                    total_loss_val = classification_loss_val+reconstruction_loss_val+ C*np.sum(np.abs(self.autoencoder(x,training=True)*ßeta))
                    #print('totel',total_loss)
                else:
                    reconstruction_loss_val = tf.reduce_mean(loss_fn2(reconstruction,x))*om
                    #print('reconstruction_loss',reconstruction_loss)
                    classification_loss_val = loss_fn1(self.model(reconstruction).reshape(-1),labels)
                    #print('classification_loss',classification_loss)
                    loss3_val= ß* tf.reduce_sum(np.abs( reconstruction))
                    #l2_val= lam* l2_reg
                    l2_val=0
                    total_loss_val = classification_loss_val+reconstruction_loss_val+ loss3_val +l2_val
                    #print('totel',total_loss_val)
            reduce_rl_plateau.on_epoch_end(epoch, total_loss_val)
            print(f'Epoch: {epoch}, '
                  f'Fine Tune Loss: {total_loss}, consits of {reconstruction_loss}, {classification_loss}, {loss3}, {l2}')
            print(f'Epoch: {epoch}, '
                  f'Fine Tune Loss: {total_loss_val}, consits of {reconstruction_loss_val}, {classification_loss_val}, {loss3_val}, {l2}')
            
            if total_loss_val> best_loss:
                trigger_times += 1
            else:
                best_model=self.autoencoder
                best_loss=total_loss_val
                trigger_times =0
            if trigger_times >= patience:
                self.autoencoder=best_model
                print('Early Stopping')
                return 
        self.autoencoder=best_model

