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

    Attention: GS and DLS only work for Python < 3.10.

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
        tsr: bool = True,
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
        self.tsr = tsr
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

    def explain(self, item, labels, TSR=None):
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
        if TSR is not None:
            self.tsr = TSR
        if self.tsr:
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

    def _call_explainer(self, current_input_data, testing_label):
        """
        Helper function to call the appropriate explainer method.
        `current_input_data` is a NumPy array, potentially batched.
        """
        # Reshape input for specific tf-explain methods if needed.
        # SHAP and FO generally don't need the trailing 1.
        if self.method == "FO":
            # FO in _getTwoStepRescaling context seems to work with (batch, S, F)
            # The original code for FO in loop: newInput (1,S,F) -> explain
            # self.NumFeatures here should be F (features)
            return self.Grad.explain(
                (current_input_data, None),
                self.model,
                class_index=testing_label,
                patch_size=self.NumFeatures,
            )
        elif self.method == "DLS" or self.method == "GS":
            # SHAP explainer (self.Grad) is already initialized.
            # Input is (batch, sequence_length, input_size)
            return np.array(self.Grad.shap_values(current_input_data))
        else:  # GRAD, IG, SG
            # These expect (batch, sequence_length, input_size, 1)
            # self.NumTimeSteps is sequence_length, self.NumFeatures is input_size
            reshaped_input = current_input_data.reshape(
                current_input_data.shape[0], self.NumTimeSteps, self.NumFeatures, 1
            )
            return self.Grad.explain(
                (reshaped_input, None), self.model, class_index=testing_label
            )

    def _getTwoStepRescaling(
        self, input_to_rescaling, TestingLabel, hasFeatureMask=None, hasSliding_window_shapes=None
    ):
        # input_to_rescaling is the data passed from explain(), could be (N,T,F) or (N,T,F,1)
        N = input_to_rescaling.shape[0] # Batch size, likely 1
        sequence_length = self.NumTimeSteps
        input_size = self.NumFeatures

        # Determine assignment value, handling potential 4D input
        if input_to_rescaling.ndim == 4:
            assignment = input_to_rescaling[0, 0, 0, 0]
        else:
            assignment = input_to_rescaling[0, 0, 0]
        
        timeGrad = np.zeros((1, sequence_length))
        inputGrad = np.zeros((input_size, 1)) # Original shape for accumulation
        newGrad = np.zeros((input_size, sequence_length))

        ActualGrad = self._call_explainer(input_to_rescaling, TestingLabel)
        # Ensure ActualGrad is consistently shaped if it comes from SHAP (which can return lists)
        if isinstance(ActualGrad, list): ActualGrad = np.array(ActualGrad[0]) # Assuming first output for multi-output models

        # Base input for modifications, ensure 3D (N, sequence_length, input_size)
        base_input_for_loops = input_to_rescaling.reshape(N, sequence_length, input_size)

        # --- Vectorized timeGrad Calculation ---
        list_newInputs_t = []
        input_swapped_template_t = base_input_for_loops.swapaxes(1, 2)  # (N, input_size, sequence_length)
        for t_idx in range(sequence_length):
            current_mod_t = input_swapped_template_t.copy()
            current_mod_t[:, :, t_idx] = assignment # Assigns to all N samples identically
            list_newInputs_t.append(current_mod_t.swapaxes(1, 2)) # Swap back to (N, sequence_length, input_size)
        
        # batched_newInput_t will have shape (N * sequence_length, sequence_length, input_size)
        # However, we want to process N samples independently if N > 1, then combine.
        # The original loop structure implies N=1 for timeGrad accumulation.
        # Let's assume N=1 for the batching logic here to match original output shapes for timeGrad.
        # If N > 1, the concatenation logic needs adjustment or results averaged.
        # For now, if N > 1, this creates N identical sets of sequence_length modifications.
        
        # Create batches for each sample in N if N > 1
        batched_newInput_t_list_per_sample = []
        for i in range(N):
            sample_base = base_input_for_loops[i:i+1, :, :] # (1, sequence_length, input_size)
            sample_input_swapped_template_t = sample_base.swapaxes(1,2) # (1, input_size, sequence_length)
            sample_list_newInputs_t = []
            for t_idx in range(sequence_length):
                current_mod_t = sample_input_swapped_template_t.copy()
                current_mod_t[0, :, t_idx] = assignment
                sample_list_newInputs_t.append(current_mod_t.swapaxes(1,2))
            batched_newInput_t_list_per_sample.append(np.concatenate(sample_list_newInputs_t, axis=0))
        
        batched_newInput_t_final = np.concatenate(batched_newInput_t_list_per_sample, axis=0)
        # Shape: (N * sequence_length, sequence_length, input_size)

        batched_timeGrad_expl = self._call_explainer(batched_newInput_t_final, TestingLabel)
        if isinstance(batched_timeGrad_expl, list): batched_timeGrad_expl = np.array(batched_timeGrad_expl[0])


        ActualGrad_repeated_t = np.repeat(ActualGrad, sequence_length, axis=0)
        time_diffs = np.absolute(ActualGrad_repeated_t - batched_timeGrad_expl)
        
        sum_axis_t = tuple(range(1, time_diffs.ndim))
        timeGrad_vector = np.sum(time_diffs, axis=sum_axis_t) # (N * sequence_length,)
        timeGrad_reshaped = timeGrad_vector.reshape(N, sequence_length)
        timeGrad = np.sum(timeGrad_reshaped, axis=0, keepdims=True) # Sum over N to get (1, sequence_length)

        timeContibution = preprocessing.minmax_scale(timeGrad, axis=1)
        meanTime = np.quantile(timeContibution, 0.55)

        # --- Loop for inputGrad and newGrad assembly ---
        for t_main in range(sequence_length):
            if timeContibution[0, t_main] > meanTime:
                list_newInputs_c_list_per_sample = []
                for i in range(N):
                    sample_base_c = base_input_for_loops[i:i+1, :, :]
                    sample_input_swapped_template_c = sample_base_c.swapaxes(1,2) # (1, input_size, sequence_length)
                    sample_list_newInputs_c = []
                    for c_idx in range(input_size):
                        current_mod_c = sample_input_swapped_template_c.copy()
                        current_mod_c[0, c_idx, t_main] = assignment
                        sample_list_newInputs_c.append(current_mod_c.swapaxes(1,2))
                    list_newInputs_c_list_per_sample.append(np.concatenate(sample_list_newInputs_c, axis=0))

                batched_newInput_c_final = np.concatenate(list_newInputs_c_list_per_sample, axis=0)
                # Shape: (N * input_size, sequence_length, input_size)

                batched_inputGrad_expl = self._call_explainer(batched_newInput_c_final, TestingLabel)
                if isinstance(batched_inputGrad_expl, list): batched_inputGrad_expl = np.array(batched_inputGrad_expl[0])

                ActualGrad_repeated_c = np.repeat(ActualGrad, input_size, axis=0) # Repeats N times for each of input_size modifications
                input_diffs = np.absolute(ActualGrad_repeated_c - batched_inputGrad_expl)

                sum_axis_c = tuple(range(1, input_diffs.ndim))
                inputGrad_vector = np.sum(input_diffs, axis=sum_axis_c) # (N * input_size,)
                inputGrad_reshaped = inputGrad_vector.reshape(N, input_size)
                # Sum over N samples to get (input_size, 1) for scaling, matching original inputGrad shape
                inputGrad_for_scaling = np.sum(inputGrad_reshaped, axis=0).reshape(input_size, 1)
                
                featureContibution = preprocessing.minmax_scale(inputGrad_for_scaling, axis=0)
            else:
                featureContibution = np.ones((input_size, 1)) * 0.1

            newGrad[:, t_main] = timeContibution[0, t_main] * featureContibution[:, 0]
        
        return np.swapaxes(newGrad, 0, 1)
