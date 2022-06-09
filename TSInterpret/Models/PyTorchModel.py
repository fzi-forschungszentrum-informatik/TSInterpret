"""Module containing an interface to trained PyTorch model."""

import torch
import numpy as np 
from TSInterpret.Models.base_model import BaseModel


class PyTorchModel(BaseModel):
    def __init__(self,model,change) -> None:
        """Init method
        :param model: trained PyTorch Model. 
        """
    #    :param model_path: path to trained model.
    #    :param backend: "PYT" for PyTorch framework.
    #    :param func: function transformation required for ML model. If func is None, then func will be the identity function.
    #    :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
    #                    dictionary of kw_args, by default.
    #    """

        super().__init__(model,change, model_path='', backend='PYT')
        #self.model = model
        #self.mode=mode
    def predict(self,item):
        '''Wrapper function for torch models.'''
        item = np.array(item.tolist(), dtype=np.float64)
        if self.change:
            item=torch.from_numpy(item.reshape(-1,item.shape[-2],item.shape[-1]))
        else: 
            item = torch.from_numpy(item)
        #if self.mode == 'time' :
        #    _ind=torch.from_numpy(item.reshape(-1,item.shape[-1],item.shape[-2]))
        out=self.model(item.float())
        y_pred = torch.nn.functional.softmax(out).detach().numpy()
        return y_pred


    #def __init__(self, model=None, model_path='', backend='PYT', func=None, kw_args=None):
    #    """Init method
    #    :param model: trained PyTorch Model.
    #    :param model_path: path to trained model.
    #    :param backend: "PYT" for PyTorch framework.
    #    :param func: function transformation required for ML model. If func is None, then func will be the identity function.
    #    :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
    #                    dictionary of kw_args, by default.
    #    """

    #    super().__init__(model, model_path, backend)

    #def load_model(self):
    #    if self.model_path != '':
    #        self.model = torch.load(self.model_path)

    #def get_output(self, input_tensor, transform_data=False):
    #    """returns prediction probabilities
    #    :param input_tensor: test input.
    #    :param transform_data: boolean to indicate if data transformation is required.
    #    """
    #    if transform_data:
    #        input_tensor = torch.tensor(self.transformer.transform(input_tensor)).float()

 #      return self.model(input_tensor).float()

    #def set_eval_mode(self):
     #   self.model.eval()

    #def get_gradient(self, input_instance):
        # Future Support
    #    raise NotImplementedError("Future Support")

    #def get_num_output_nodes(self, inp_size):
    #    temp_input = torch.rand(1, inp_size).float()
    #    return self.get_output(temp_input).data