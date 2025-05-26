import numpy as np
import torch
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    FeatureAblation,
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
    ShapleyValueSampling,
)
from sklearn import preprocessing
from torch.autograd import Variable

from TSInterpret.InterpretabilityModels.Saliency.Saliency_Base import Saliency as Sal


class Saliency_PTY(Sal):
    """
    PyTorch Implementation for Saliency Calculation based on [1]. The Saliency Methods are based on the library captum [2].
    For PyTorch the following saliency methods are available:
        + Gradients (GRAD)
        + Integrated Gradients (IG)
        + Gradient Shap (GS)
        + DeepLift (DL)
        + DeepLiftShap (DLS)
        + SmoothGrad (SG)
        + Shapley Value Sampling(SVS)
        + Feature Ablatiom (FA)
        + Occlusion (FO)
    References
    ----------
    [1] Ismail, Aya Abdelsalam, et al.
    "Benchmarking deep learning interpretability in time series predictions."
    Advances in neural information processing systems 33 (2020): 6441-6452.
    [2] Kokhlikyan, Narine, et al.
    "Captum: A unified and generic model interpretability library for pytorch."
    arXiv preprint arXiv:2009.07896 (2020).
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
        normalize:bool=True,
        device: str = "cpu",
    ) -> None:
        """Initialization
        Arguments:
            model [torch.nn.Module]: model to be explained
            NumTimeSteps int : Number of Time Step
            NumFeatures int : Number Features
            method str: Saliency Methode to be used
            mode str: Second dimension 'time'->`(1,time,feat)`  or 'feat'->`(1,feat,time)`
        """
        super().__init__(model, NumTimeSteps, NumFeatures, method, mode,normalize)
        self.method = method
        self.tsr = tsr
        #self.normalize=normalize
        if method == "GRAD":
            self.Grad = Saliency(model)
        elif method == "IG":
            self.Grad = IntegratedGradients(model)
        elif method == "GS":
            self.Grad = GradientShap(model)
        elif method == "DL":
            self.Grad = DeepLift(model)
        elif method == "DLS":
            self.Grad = DeepLiftShap(model)
        elif method == "SG":
            Grad_ = Saliency(model)
            self.Grad = NoiseTunnel(Grad_)
        elif method == "SVS":
            self.Grad = ShapleyValueSampling(model)
        # elif method == 'FP':
        #    self.Grad = FeaturePermutation(model)
        elif method == "FA":
            self.Grad = FeatureAblation(model)
        elif method == "FO":
            self.Grad = Occlusion(model)
        self.device = device

    def explain(self, item: np.ndarray, labels: int, TSR=None, **kwargs):
        """Method to explain the model based on the item.
        Arguments:
            item np.array: item to get feature attribution for, if `mode = time`->`(1,time,feat)`  or `mode = feat`->`(1,feat,time)`
            labels int: label
            TSR bool: if True time series rescaling according to [1] is used, else plain (scaled) weights are returened
        Returns:
            np.array: feature attribution weights `mode = time`->`(time,feat)` or `mode = feat`->`(feat,time)`
        """
        try:
            labels = int(labels)
        except ValueError:
            raise Exception("Please provide the labels as int.")
        
        # Convert item to PyTorch tensor early if it's a NumPy array
        if isinstance(item, np.ndarray):
            input_tensor = torch.from_numpy(item.copy()).float() # Use .copy() to avoid issues with writeability
        else:
            input_tensor = item.float()

        # Reshape input_tensor based on mode BEFORE sending to device and Variable
        # Captum expects (N, ..., F) or (N, ..., T) based on what features represent
        # If mode=='feat', input (N,F,T). If mode=='time', input (N,T,F)
        # The original code had a swap for 'feat' mode which is unusual.
        # Sticking to: 'time' -> (N,T,F), 'feat' -> (N,F,T) as per docstring
        # No, the original code has: `if self.mode=='feat': input = np.swapaxes(input, -1, -2)`
        # This means if mode is 'feat', (N,F,T) becomes (N,T,F).
        # This is counter-intuitive but let's replicate.
        if self.mode == 'feat':
            input_tensor = input_tensor.permute(0, 2, 1) # (N,F,T) -> (N,T,F)

        input_tensor = Variable(input_tensor, requires_grad=True).to(self.device)
        
        # batch_size = input_tensor.shape[0] # Not used in current refactor of this function

        # Mask related logic - ensure shapes are consistent with input_tensor
        # The original loop for mask creation:
        # mask = np.zeros((self.NumTimeSteps, self.NumFeatures), dtype=int) # This is (T,F)
        # for i in range(self.NumTimeSteps):
        #    mask[i, :] = i
        # Is replaced by the vectorized version below:
        mask = np.tile(np.arange(self.NumTimeSteps), (self.NumFeatures, 1)).T
        # The following lines related to mask_single remain commented as per their state in the input file.
        # Their functionality depends on how `mask` is intended to be used for SVS or default inputMask.
        # mask_single = torch.from_numpy(mask).to(self.device) # (T,F)
        # mask_single = mask_single.reshape(1, self.NumTimeSteps, self.NumFeatures) # (1,T,F)

        # inputMask handling from kwargs
        inputMask = None # Default to None
        if "inputMask" in kwargs:
            inputMask = kwargs["inputMask"]
            if isinstance(inputMask, np.ndarray):
                inputMask = torch.from_numpy(inputMask).to(self.device)
        # else:
            # Default mask creation was complex and potentially mismatched input_tensor shape after permute
            # For now, if not provided, it's None. SVS might need a specific default.
            # mask_shape = input_tensor.shape[1:] # e.g. (T,F) or (F,T)
            # default_mask_np = np.zeros((1,) + mask_shape, dtype=int) # (1, D1, D2)
            # This needs more context on how default mask should be structured.
            # For simplicity, relying on user-provided mask or method-specific defaults in Captum.

        # Baselines
        baseline_dims = input_tensor.shape
        if "baseline_single" in kwargs:
            baseline_single = kwargs["baseline_single"]
            if isinstance(baseline_single, np.ndarray):
                baseline_single = torch.from_numpy(baseline_single).float().to(self.device)
        else:
            baseline_single = torch.randn(baseline_dims).float().to(self.device) # Changed from random to randn

        if "baseline_multiple" in kwargs:
            baseline_multiple = kwargs["baseline_multiple"]
            if isinstance(baseline_multiple, np.ndarray):
                baseline_multiple = torch.from_numpy(baseline_multiple).float().to(self.device)
        else:
            # Ensure first dim is multiple of input_tensor.shape[0] if that's Captum's expectation
            # Or simply create (num_samples, D1, D2, ...)
            num_baseline_samples = input_tensor.shape[0] * 5 # Or a fixed number like 25
            baseline_multiple_shape = (num_baseline_samples,) + baseline_dims[1:]
            baseline_multiple = torch.randn(baseline_multiple_shape).float().to(self.device)

        # Common arguments for attribute calls
        attribute_kwargs = {'target': labels}
        base_for_tsr = None # Stores the baseline used, for TSR
        sliding_window_for_tsr = None # Stores sliding window, for TSR

        if self.method == "GRAD":
            attributions = self.Grad.attribute(input_tensor, **attribute_kwargs)
        elif self.method in ["IG", "DL"]:
            base_for_tsr = baseline_single
            attribute_kwargs['baselines'] = base_for_tsr
            attributions = self.Grad.attribute(input_tensor, **attribute_kwargs)
        elif self.method in ["GS", "DLS"]:
            base_for_tsr = baseline_multiple
            attribute_kwargs['baselines'] = base_for_tsr
            if self.method == "GS":
                attribute_kwargs['stdevs'] = 0.09 # Typical for GradientShap
            attributions = self.Grad.attribute(input_tensor, **attribute_kwargs)
        elif self.method == "SG":
            # SmoothGrad specific: nt_samples, nt_type, stdevs
            # Example: attribute_kwargs.update({'nt_samples': 10, 'stdevs': 0.1})
            attributions = self.Grad.attribute(input_tensor, **attribute_kwargs)
        elif self.method == "SVS":
            base_for_tsr = baseline_single
            attribute_kwargs['baselines'] = base_for_tsr
            if inputMask is not None: # SVS uses feature_mask
                 attribute_kwargs['feature_mask'] = inputMask
            attributions = self.Grad.attribute(input_tensor, **attribute_kwargs)
        elif self.method == "FA":
            attributions = self.Grad.attribute(input_tensor, **attribute_kwargs)
        elif self.method == "FO":
            base_for_tsr = baseline_single
            attribute_kwargs['baselines'] = base_for_tsr
            # Occlusion sliding window: (num_features_dim1, num_features_dim2, ...)
            # Assuming features are the last dimension after potential permute by mode
            # If input_tensor is (N,T,F), feature dim is -1. If (N,F,T), feature dim is -2.
            # self.NumFeatures refers to one of these.
            # This needs to be set carefully based on input_tensor's current shape.
            # For (N, D1, D2), if features are along D2, sliding_window = (1, k) or (k,1) if features along D1
            # The original code:
            # if self.mode == "feat": sliding_window_for_tsr = (1, self.NumFeatures) -> input (N,T,F), occlude F
            # else: sliding_window_for_tsr = (self.NumFeatures, 1) -> input (N,T,F) originally, occlude F
            # This seems to imply features are always the last dim for FO's perspective.
            # Let's assume features are the last dimension of input_tensor for FO.
            sliding_window_for_tsr = tuple([1] * (input_tensor.ndim - 2) + [self.NumFeatures if self.NumFeatures > 0 else 1])
            attribute_kwargs['sliding_window_shapes'] = sliding_window_for_tsr
            attributions = self.Grad.attribute(input_tensor, **attribute_kwargs)
        
        if TSR is not None:
            self.tsr = TSR

        if self.tsr:
            assignment_val = kwargs.get("assignment", None) # Use .get for safety
            
            tsr_call_kwargs = {'hasBaseline': base_for_tsr, 
                               'hasSliding_window_shapes': sliding_window_for_tsr}
            if self.method == "SVS" and inputMask is not None:
                 tsr_call_kwargs['hasFeatureMask'] = inputMask

            TSR_attributions = self._getTwoStepRescaling(
                input_tensor, # Pass the tensor that was used for ActualGrad calculation
                labels,
                assignment=assignment_val,
                **tsr_call_kwargs
            )
            TSR_saliency = self._givenAttGetRescaledSaliency(
                TSR_attributions, isTensor=False # TSR_attributions is numpy array
            )
            return TSR_saliency # Expected (T,F) or (F,T) if N=1
        else:
            saliency_val = self._givenAttGetRescaledSaliency(attributions, isTensor=True) # attributions is Tensor
            return saliency_val[0] # Return first sample if batched, typically (T,F) or (F,T)


    def _getTwoStepRescaling(
        self,
        input_tensor_orig, # Tensor used for ActualGrad, shape depends on self.mode + permute in explain
        TestingLabel,
        hasBaseline=None,
        hasFeatureMask=None,
        hasSliding_window_shapes=None,
        assignment=None,
    ):
        N = input_tensor_orig.shape[0] # Batch size

        # Determine canonical shape (N, F, T) for internal loop processing
        # sequence_length = T (time steps), input_size = F (features)
        if self.mode == "time": # input_tensor_orig is (N, T, F) due to explain() logic for 'time'
            # Canonical for loops: (N, F, T)
            input_for_loops = input_tensor_orig.permute(0, 2, 1)
            sequence_length = self.NumTimeSteps # T
            input_size = self.NumFeatures      # F
        else: # self.mode == "feat", input_tensor_orig is (N, T, F) due to explain() permute for 'feat'
              # This means original item was (N,F,T). For loops, we want (N,F,T).
              # So input_tensor_orig (N,T,F) must be permuted back to (N,F,T).
            input_for_loops = input_tensor_orig.permute(0, 2, 1) # (N,T,F) -> (N,F,T)
            sequence_length = self.NumTimeSteps # T (from original F dim)
            input_size = self.NumFeatures      # F (from original T dim)
            # This is confusing. Let's simplify:
            # NumTimeSteps is always time, NumFeatures is always features.
            # input_for_loops should be (N, self.NumFeatures, self.NumTimeSteps)
            # If self.mode == 'time', input_tensor_orig is (N, T, F). Permute to (N,F,T).
            # If self.mode == 'feat', input_tensor_orig is (N, T, F) (item was N,F,T then permuted). Permute to (N,F,T).
            # So the permute is the same in both cases if input_tensor_orig is (N,T,F)
            # This means input_tensor_orig should be shaped (N, T, F) when passed here.
            # Let's re-verify explain():
            # if self.mode == 'feat': input_tensor = input_tensor.permute(0, 2, 1) # (N,F,T) -> (N,T,F)
            # So, input_tensor_orig is (N,T,F) if mode=feat, and (N,T,F) if mode=time.
            # Thus, input_for_loops = input_tensor_orig.permute(0,2,1) makes it (N,F,T)
            input_for_loops = input_tensor_orig.permute(0,2,1) # Now (N, self.NumFeatures, self.NumTimeSteps)
            sequence_length = self.NumTimeSteps
            input_size = self.NumFeatures


        if assignment is None:
            assignment = input_for_loops[0, 0, 0].clone().detach()
        else: # Ensure assignment is a tensor of correct type and device
            if not isinstance(assignment, torch.Tensor):
                assignment = torch.tensor(assignment, dtype=input_for_loops.dtype, device=input_for_loops.device)
            else:
                assignment = assignment.to(input_for_loops.dtype).to(input_for_loops.device)


        actual_grad_attr_kwargs = {'target': TestingLabel}
        if hasBaseline is not None: actual_grad_attr_kwargs['baselines'] = hasBaseline
        if hasFeatureMask is not None: actual_grad_attr_kwargs['feature_mask'] = hasFeatureMask
        if hasSliding_window_shapes is not None: actual_grad_attr_kwargs['sliding_window_shapes'] = hasSliding_window_shapes
        
        ActualGrad = self.Grad.attribute(input_tensor_orig, **actual_grad_attr_kwargs).data.cpu().numpy()
        ActualGrad_torch = torch.from_numpy(ActualGrad).to(input_for_loops.device)
        
        list_inputs_for_timeGrad_attr = []
        for t_idx in range(sequence_length):
            newInput_t = input_for_loops.clone()
            newInput_t[:, :, t_idx] = assignment # Assign to all features at time t_idx
            
            # Map back to input_tensor_orig's domain for attribute call
            # input_tensor_orig is (N,T,F). newInput_t is (N,F,T). So permute.
            input_to_attr_t = newInput_t.permute(0, 2, 1) 
            list_inputs_for_timeGrad_attr.append(input_to_attr_t)

        stacked_inputs_t = torch.cat(list_inputs_for_timeGrad_attr, dim=0)

        time_attr_kwargs = {'target': TestingLabel}
        if hasBaseline is not None: time_attr_kwargs['baselines'] = hasBaseline.repeat_interleave(sequence_length, dim=0)
        if hasFeatureMask is not None: time_attr_kwargs['feature_mask'] = hasFeatureMask.repeat_interleave(sequence_length, dim=0)
        if hasSliding_window_shapes is not None: time_attr_kwargs['sliding_window_shapes'] = hasSliding_window_shapes # Fixed tuple, no repeat needed

        attr_result_t_batched = self.Grad.attribute(stacked_inputs_t, **time_attr_kwargs).data.cpu().numpy()
        ActualGrad_repeated_t = ActualGrad_torch.repeat_interleave(sequence_length, dim=0).cpu().numpy()
        time_diffs = np.absolute(ActualGrad_repeated_t - attr_result_t_batched)

        # time_diffs is (N*T_seq, D1_orig, D2_orig). D1,D2 from input_tensor_orig (e.g., T,F)
        # For summation, we need a canonical (F,T) view of contribution per (N, t_idx)
        # If input_tensor_orig was (N,T,F), time_diffs are (N*T_seq, T, F). Swap to (N*T_seq, F, T).
        time_diffs_canonical_sum_shape = time_diffs.swapaxes(-1,-2) # Now (N*T_seq, F, T)

        timeGrad_summed = np.sum(time_diffs_canonical_sum_shape, axis=tuple(range(1, time_diffs_canonical_sum_shape.ndim)))
        timeGrad_reshaped = timeGrad_summed.reshape(N, sequence_length)
        timeGrad = np.sum(timeGrad_reshaped, axis=0, keepdims=True)

        timeContribution = preprocessing.minmax_scale(timeGrad, axis=1)
        meanTime = np.quantile(timeContribution, 0.55)

        newGrad = np.zeros((input_size, sequence_length)) # (F, T)

        if input_size > 1:
            for t_loop_idx in range(sequence_length):
                if timeContribution[0, t_loop_idx] > meanTime:
                    list_inputs_for_featGrad_attr = []
                    for c_idx in range(input_size):
                        newInput_c = input_for_loops.clone() # (N, F, T)
                        newInput_c[:, c_idx, t_loop_idx] = assignment
                        input_to_attr_c = newInput_c.permute(0, 2, 1) # Map to (N,T,F) for attribute
                        list_inputs_for_featGrad_attr.append(input_to_attr_c)
                    
                    stacked_inputs_c = torch.cat(list_inputs_for_featGrad_attr, dim=0)

                    feat_attr_kwargs = {'target': TestingLabel}
                    if hasBaseline is not None: feat_attr_kwargs['baselines'] = hasBaseline.repeat_interleave(input_size, dim=0)
                    if hasFeatureMask is not None: feat_attr_kwargs['feature_mask'] = hasFeatureMask.repeat_interleave(input_size, dim=0)
                    if hasSliding_window_shapes is not None: feat_attr_kwargs['sliding_window_shapes'] = hasSliding_window_shapes

                    attr_result_c_batched = self.Grad.attribute(stacked_inputs_c, **feat_attr_kwargs).data.cpu().numpy()
                    ActualGrad_repeated_c = ActualGrad_torch.repeat_interleave(input_size, dim=0).cpu().numpy()
                    
                    feat_diffs = np.absolute(ActualGrad_repeated_c - attr_result_c_batched)
                    # Original code: inputGrad_perInput = np.swapaxes(inputGrad_perInput, -1, -2) unconditionally
                    # feat_diffs is (N*F, T, F). Swap to (N*F, F, T)
                    feat_diffs_swapped = feat_diffs.swapaxes(-1, -2) 
                    
                    inputGrad_summed = np.sum(feat_diffs_swapped, axis=tuple(range(1, feat_diffs_swapped.ndim)))
                    inputGrad_reshaped = inputGrad_summed.reshape(N, input_size)
                    inputGrad_for_t_calc = np.sum(inputGrad_reshaped, axis=0).reshape(input_size,1)
                    
                    featureContribution = preprocessing.minmax_scale(inputGrad_for_t_calc, axis=0)
                else:
                    featureContribution = np.ones((input_size, 1)) * 0.1
                
                newGrad[:, t_loop_idx] = timeContribution[0, t_loop_idx] * featureContribution[:, 0]
        else: 
            newGrad = timeContribution.copy() # timeContribution is (1,T), newGrad is (F,T) -> (1,T)

        # Final swap based on self.mode for output consistency
        # newGrad is (F,T). If mode was 'time', original output was (T,F).
        if self.mode == "time":
            newGrad = np.swapaxes(newGrad, -1, -2) # (F,T) -> (T,F)
            
        return newGrad

    def _givenAttGetRescaledSaliency(self, attributions, isTensor=True):
        if isTensor:
            saliency_np = np.absolute(attributions.detach().cpu().numpy())
        else:
            saliency_np = np.absolute(attributions)
        
        N = saliency_np.shape[0]
        original_shape = saliency_np.shape
        
        saliency_flat = saliency_np.reshape(N, -1)
        
        # Handle cases where all values in a row are the same (avoid division by zero in minmax_scale)
        min_vals = np.min(saliency_flat, axis=1, keepdims=True)
        max_vals = np.max(saliency_flat, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        
        rescaled_saliency_flat = np.zeros_like(saliency_flat)
        # Scale only where range is not zero
        non_zero_range_mask = (range_vals != 0).squeeze()
        if np.any(non_zero_range_mask):
             rescaled_saliency_flat[non_zero_range_mask,:] = \
                (saliency_flat[non_zero_range_mask,:] - min_vals[non_zero_range_mask,:]) / \
                 range_vals[non_zero_range_mask,:]
        # For rows where all values were same, default to 0 or 0.5. Original was 0.
        # If min_val = max_val, then (X - min_val) is 0. So result is 0. This is fine.

        rescaledSaliency = rescaled_saliency_flat.reshape(original_shape)
        return rescaledSaliency
