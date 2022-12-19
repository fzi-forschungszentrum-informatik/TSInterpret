import tensorflow as tf

from TSInterpret.Models.base_model import BaseModel


class TensorFlowModel(BaseModel):
    def __init__(self, model, change=False) -> None:
        """Wrapper for Tensorflow Models that unifiy the prediction function for a classifier.
        Arguments:
            model : Trained TF Model.
            change bool: if swapping of dimension is necessary = True
        """
        super().__init__(model, change, model_path="", backend="TF")

    def predict(self, item):
        """Unified prediction function.
        Arguments:
            item np.array: item to be classified.
         Returns:
            an array of output scores for a classifier.
        """
        if self.change:
            item = item.reshape(item.shape[0], item.shape[2], item.shape[1])
        out = self.model.predict(item)
        return out

    def load_model(self, path):
        return super().load_model(path)

    def get_num_output_nodes(self, inp_size):
        temp_input = tf.convert_to_tensor(
            [tf.random.uniform([inp_size])], dtype=tf.float32
        )
        return self.get_output(temp_input)
