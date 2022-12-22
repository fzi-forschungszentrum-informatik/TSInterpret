import tensorflow
import torch

from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_PTY import Saliency_PTY
from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_TF import Saliency_TF


class TSR:
    """
    Wrapper Class for Saliency Calculation.
    Automatically calls the corresponding PYT or TF implementation.
    """

    def __new__(
        self, model, NumTimeSteps, NumFeatures, method="GRAD", mode="time", device="cpu"
    ):
        """Initialization
        Arguments:
            model [torch.nn.Module, tf.keras.Model]: model to be explained
            NumTimeSteps int : Number of Time Step
            NumFeatures int : Number Features
            method str: Saliency Methode to be used
            mode str: Second dimension 'time'->`(1,time,feat)`  or 'feat'->`(1,feat,time)`
        """
        if isinstance(model, torch.nn.Module):

            return Saliency_PTY(
                model,
                NumTimeSteps,
                NumFeatures,
                method=method,
                mode=mode,
                device=device,
            )

        elif isinstance(model, tensorflow.keras.Model):

            return Saliency_TF(
                model, NumTimeSteps, NumFeatures, method=method, mode=mode
            )
        else:
            raise NotImplementedError(
                "Please use a TF or PYT Classification model! \
                If the current model is a TF or PYT Model, \
                try calling the wrappers directly \
                (TF -> Saliency_TF, PYT -> Saliency_PYT)"
            )
