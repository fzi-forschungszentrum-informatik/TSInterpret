import numpy as np
import shap
from sklearn import preprocessing

# from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.integrated_gradients import IntegratedGradients
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.smoothgrad import SmoothGrad
from tf_explain.core.vanilla_gradients import VanillaGradients

from TSInterpret.InterpretabilityModels.Saliency.Saliency_Base import Saliency


class Saliency_TF(Saliency):
    """
    Tensorflow Implementation for Saliency Calculation based on [1].
    The Saliency Methods are based on the library tf-explain [2] and shap [3].
    For Tensorflow the following saliency methods are available:
        + Gradients (GRAD)
        + Integrated Gradients (IG)
        + Gradient Shap (GS))
        + DeepLiftShap (DLS)
        + SmoothGrad (SG)
        + Occlusion (FO)
    References
    ----------
    [1] Ismail, Aya Abdelsalam, et al.
    "Benchmarking deep learning interpretability in time series predictions."
    Advances in neural information processing systems 33 (2020): 6441-6452.

    [2] Meudec, Raphael: , tf-explain. https://github.com/sicara/tf-explain

    [3] Lundberg, Scott M., and Su-In Lee.
    "A unified approach to interpreting model predictions."
    Advances in neural information processing systems 30 (2017).
        https://shap.readthedocs.io/
    ----------
    """

    def __init__(
        self,
        model,
        NumTimeSteps: int,
        NumFeatures: int,
        method: str = "GRAD",
        mode: str = "time",
        device: str = "cpu",
    ) -> None:
        """
        Arguments:
            model [tf.keras.models]: model to be explained
            NumTimeSteps int : Number of Time Step
            NumFeatures int : Number Features
            method str: Saliency Methode to be used
            mode str: Second dimension 'time'->`(1,time,feat)`  or 'feat'->`(1,feat,time)`
        """
        # tf explain does not provide baseline !
        super().__init__(model, NumTimeSteps, NumFeatures, method, mode)
        print("Mode in TF Saliency", self.mode)
        self.method = method
        if method == "GRAD":
            self.Grad = VanillaGradients()
        if method == "IG":
            self.Grad = IntegratedGradients()
        # elif method == 'DL':
        #    self.Grad = DeepLift(model)
        elif method == "DLS":
            self.Grad = shap.DeepExplainer
        elif method == "GS":
            self.Grad = shap.GradientExplainer
        elif self.method == "SG":
            self.Grad = SmoothGrad()

        # elif method == 'SVS':
        #    self.Grad = ShapleyValueSampling(model)
        # elif method == 'FP':
        #    self.Grad = FeaturePermutation(model)
        # elif method == 'FA':
        #    self.Grad = FeatureAblation(model)
        elif method == "FO":
            self.Grad = OcclusionSensitivity()

    def explain(self, item, labels, TSR=True):
        """Method to explain the model based on the item.
        Arguments:
            item np.array: item to get feature attribution for, if `mode = time`->`(1,time,feat)`  or `mode = feat`->`(1,feat,time)`
            labels int: label
            TSR bool: if True time series rescaling according to [1] is used, else plain (scaled) weights are returened
        Returns:
            np.array: feature attribution weights `mode = time`->`(time,feat)` or `mode = feat`->`(feat,time)`

        """
        rescaledGrad = np.zeros(item.shape)
        idx = 0
        input = item.reshape(-1, self.NumFeatures, self.NumTimeSteps)

        batch_size = input.shape[0]

        input = input.reshape(-1, self.NumTimeSteps, self.NumFeatures)
        if self.method == "IG" or self.method == "GRAD" or self.method == "SG":
            input = input.reshape(-1, self.NumTimeSteps, self.NumFeatures, 1)
            attributions = self.Grad.explain(
                (input, None), self.model, class_index=labels
            )
        elif self.method == "DLS" or self.method == "GS":
            self.Grad = self.Grad(self.model, input)
            attributions = self.Grad.shap_values(input)[0]
        elif self.method == "FO":
            input = input.reshape(-1, self.NumFeatures, self.NumTimeSteps, 1)
            attributions = self.Grad.explain(
                (input, None),
                self.model,
                class_index=labels,
                patch_size=self.NumFeatures,
            )

        if TSR:
            # print(base)
            TSR_attributions = self._getTwoStepRescaling(input, labels)
            TSR_saliency = self._givenAttGetRescaledSaliency(TSR_attributions)
            return TSR_saliency
        else:
            rescaledGrad[
                idx : idx + batch_size, :, :
            ] = self._givenAttGetRescaledSaliency(attributions)
            return np.array(rescaledGrad[0])

    def _givenAttGetRescaledSaliency(self, attributions):
        saliency = np.absolute(attributions)
        saliency = saliency.reshape(-1, self.NumTimeSteps * self.NumFeatures)
        rescaledSaliency = preprocessing.minmax_scale(saliency, axis=1)
        rescaledSaliency = rescaledSaliency.reshape(np.array(attributions).shape)
        return rescaledSaliency

    def _getTwoStepRescaling(
        self, input, TestingLabel, hasFeatureMask=None, hasSliding_window_shapes=None
    ):
        sequence_length = self.NumTimeSteps
        input_size = self.NumFeatures
        assignment = input[0, 0, 0]
        timeGrad = np.zeros((1, sequence_length))
        inputGrad = np.zeros((input_size, 1))
        newGrad = np.zeros((input_size, sequence_length))
        if self.method == "FO":
            ActualGrad = self.Grad.explain(
                (input, None),
                self.model,
                class_index=TestingLabel,
                patch_size=self.NumFeatures,
            )
        elif self.method == "DLS" or self.method == "GS":
            ActualGrad = self.Grad.shap_values(input)
            ActualGrad = np.array(ActualGrad)
            # print(ActualGrad.shape)
        else:
            ActualGrad = self.Grad.explain(
                (input, None), self.model, class_index=TestingLabel
            )  # .data.cpu().numpy()

        for t in range(sequence_length):
            newInput = input.copy().reshape(1, input_size, sequence_length)
            newInput[:, :, t] = assignment
            newInput = newInput.reshape(1, sequence_length, input_size)
            if self.method == "FO":
                timeGrad_perTime = self.Grad.explain(
                    (newInput, None),
                    self.model,
                    class_index=TestingLabel,
                    patch_size=self.NumFeatures,
                )
            elif self.method == "DLS" or self.method == "GS":
                timeGrad_perTime = self.Grad.shap_values(newInput)
                timeGrad_perTime = np.array(timeGrad_perTime)
            else:
                newInput = newInput.reshape(1, sequence_length, input_size, 1)
                timeGrad_perTime = self.Grad.explain(
                    (newInput, None), self.model, class_index=TestingLabel
                )  # .data.cpu().numpy()
            timeGrad_perTime = np.absolute(ActualGrad - timeGrad_perTime)
            timeGrad[:, t] = np.sum(timeGrad_perTime)

        timeContibution = preprocessing.minmax_scale(timeGrad, axis=1)
        meanTime = np.quantile(timeContibution, 0.55)

        for t in range(sequence_length):
            if timeContibution[0, t] > meanTime:
                for c in range(input_size):
                    newInput = input.copy().reshape(1, input_size, sequence_length)
                    newInput[:, c, t] = assignment
                    newInput = newInput.reshape(1, sequence_length, input_size)
                    if self.method == "FO":
                        inputGrad_perInput = self.Grad.explain(
                            (newInput, None),
                            self.model,
                            class_index=TestingLabel,
                            patch_size=self.NumFeatures,
                        )
                    elif self.method == "DLS" or self.method == "GS":
                        inputGrad_perInput = self.Grad.shap_values(newInput)
                        inputGrad_perInput = np.array(inputGrad_perInput)
                        # print(inputGrad_perInput.shape)
                    else:
                        newInput = newInput.reshape(1, sequence_length, input_size, 1)
                        inputGrad_perInput = self.Grad.explain(
                            (newInput, None), self.model, class_index=TestingLabel
                        )  # .data.cpu().numpy()

                    inputGrad_perInput = np.absolute(ActualGrad - inputGrad_perInput)
                    inputGrad[c, :] = np.sum(inputGrad_perInput)

                featureContibution = preprocessing.minmax_scale(inputGrad, axis=0)
            else:
                featureContibution = np.ones((input_size, 1)) * 0.1

            for c in range(input_size):
                newGrad[c, t] = timeContibution[0, t] * featureContibution[c, 0]
        return newGrad.reshape(sequence_length, input_size)
