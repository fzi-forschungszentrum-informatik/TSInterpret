import tensorflow as tf
from tensorflow import keras

from Models.base_model import BaseModel


class TensorFlowModel(BaseModel):

    def __init__(self,model,change) -> None:
        super().__init__(model,change, model_path='', backend='TF2')
        #self.model = model
        #self.mode = mode

    def predict(self,item):
        #print(item.shape)
        #print(self.mode)
        if self.change:
            item=item.reshape(item.shape[0],item.shape[2],item.shape[1])
        #if self.mode == 'feat' :
        #    print('Feat')
        #    item=item.reshape(item.shape[0],item.shape[1],item.shape[2])
        #print(item.shape)
        out=self.model.predict(item)
        return out

    #def __init__(self, model=None, model_path='', backend='TF1', func=None, kw_args=None):
    #    """Init method
    #    :param model: trained Keras Sequential Model.
    #    :param model_path: path to trained model.
    #    :param backend: "TF1" for TensorFlow 1 and "TF2" for TensorFlow 2.
    #    :param func: function transformation required for ML model. If func is None, then func will be the identity function.
    #    :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
    #                    dictionary of kw_args, by default.
    #    """

    #    super().__init__(model, model_path, backend)

    #def load_model(self):
    #    if self.model_path != '':
    #        self.model = keras.models.load_model(self.model_path)

    #def get_output(self, input_tensor, training=False, transform_data=False):
    #    """returns prediction probabilities
    #    :param input_tensor: test input.
    #    :param training: to determine training mode in TF2.
    #    :param transform_data: boolean to indicate if data transformation is required.
     #   """
    #    if transform_data:
    #        input_tensor = tf.constant(self.transformer.transform(input_tensor), dtype=tf.float32)#

    #    if self.backend == 'TF2':
    #        return self.model(input_tensor, training=training)
    #    else:
    #        return self.model(input_tensor)

    #def get_gradient(self, input_instance):
        # Future Support
    #    raise NotImplementedError("Future Support")

    def get_num_output_nodes(self, inp_size):
        temp_input = tf.convert_to_tensor([tf.random.uniform([inp_size])], dtype=tf.float32)
        return self.get_output(temp_input)