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
        super().__init__(model, NumTimeSteps, NumFeatures, method, mode)
        self.method = method
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

    def explain(self, item: np.ndarray, labels: int, TSR: bool = True):
        """Method to explain the model based on the item.
        Arguments:
            item np.array: item to get feature attribution for, if `mode = time`->`(1,time,feat)`  or `mode = feat`->`(1,feat,time)`
            labels int: label
            TSR bool: if True time series rescaling according to [1] is used, else plain (scaled) weights are returened
        Returns:
            np.array: feature attribution weights `mode = time`->`(time,feat)` or `mode = feat`->`(feat,time)`
        """
        mask = np.zeros((self.NumTimeSteps, self.NumFeatures), dtype=int)
        for i in range(self.NumTimeSteps):
            mask[i, :] = i
        rescaledGrad = np.zeros(item.shape)
        idx = 0
        item = np.array(item.tolist(), dtype=np.float64)
        input = torch.from_numpy(item)

        input = input.reshape(-1, self.NumTimeSteps, self.NumFeatures).to(self.device)
        input = Variable(input, volatile=False, requires_grad=True)

        batch_size = input.shape[0]

        inputMask = np.zeros(input.shape)
        inputMask[:, :, :] = mask
        inputMask = torch.from_numpy(inputMask).to(self.device)
        mask_single = torch.from_numpy(mask).to(self.device)
        mask_single = mask_single.reshape(1, self.NumTimeSteps, self.NumFeatures).to(
            self.device
        )
        # input = samples.reshape(-1, args.NumTimeSteps, args.NumFeatures).to(device)
        input = input.reshape(-1, self.NumFeatures, self.NumTimeSteps)
        baseline_single = (
            torch.from_numpy(np.random.random(input.shape)).float().to(self.device)
        )
        baseline_multiple = (
            torch.from_numpy(
                np.random.random((input.shape[0] * 5, input.shape[1], input.shape[2]))
            )
            .float()
            .to(self.device)
        )
        input = input.float()
        base = None
        has_sliding_window = None
        if self.method == "GRAD":
            attributions = self.Grad.attribute(input, target=labels)
        elif self.method == "IG":
            base = baseline_single
            attributions = self.Grad.attribute(
                input, baselines=baseline_single, target=labels
            )
        elif self.method == "DL":
            base = baseline_single
            attributions = self.Grad.attribute(
                input, baselines=baseline_single, target=labels
            )
        elif self.method == "GS":
            base = baseline_multiple
            attributions = self.Grad.attribute(
                input, baselines=baseline_multiple, stdevs=0.09, target=labels
            )
        elif self.method == "DLS":
            base = baseline_multiple
            attributions = self.Grad.attribute(
                input, baselines=baseline_multiple, target=labels
            )
        elif self.method == "SG":
            attributions = self.Grad.attribute(input, target=labels)
        elif self.method == "SVS":
            base = baseline_single
            inputMask = inputMask.reshape(
                -1, baseline_single.shape[1], baseline_single.shape[2]
            )
            attributions = self.Grad.attribute(
                input, baselines=baseline_single, target=labels, feature_mask=inputMask
            )
        # elif(self.method=='FP'):
        #    attributions = self.Grad.attribute(input, target=labels, perturbations_per_eval= input.shape[0],feature_mask=mask_single)
        elif self.method == "FA":
            attributions = self.Grad.attribute(input, target=labels)
        elif self.method == "FO":
            base = baseline_single
            has_sliding_window = (1, self.NumFeatures)
            attributions = self.Grad.attribute(
                input,
                sliding_window_shapes=(1, self.NumFeatures),
                target=labels,
                baselines=baseline_single,
            )

        if TSR:
            # print('TSR', TSR)
            TSR_attributions = self._getTwoStepRescaling(
                input,
                labels,
                hasBaseline=base,
                hasSliding_window_shapes=has_sliding_window,
            )
            # print('TSR Attribution', TSR_attributions.shape)
            TSR_saliency = self._givenAttGetRescaledSaliency(
                TSR_attributions, isTensor=False
            )
            # print('TSR Saliency', TSR_saliency.shape)
            return TSR_saliency
        else:
            # print('TSR', TSR)
            # TODO attributions does not exist for SVS and Fo
            rescaledGrad[
                idx : idx + batch_size, :, :
            ] = self._givenAttGetRescaledSaliency(attributions)
            # print('Rescaled', rescaledGrad.shape)
            return rescaledGrad[0]

    def _getTwoStepRescaling(
        self,
        input,
        TestingLabel,
        hasBaseline=None,
        hasFeatureMask=None,
        hasSliding_window_shapes=None,
    ):
        sequence_length = self.NumTimeSteps
        input_size = self.NumFeatures
        assignment = input[0, 0, 0]
        timeGrad = np.zeros((1, sequence_length))
        inputGrad = np.zeros((input_size, 1))
        newGrad = np.zeros((input_size, sequence_length))
        # print("has Sliding Window", hasSliding_window_shapes)
        if hasBaseline is None:
            ActualGrad = (
                self.Grad.attribute(input, target=TestingLabel).data.cpu().numpy()
            )
        else:
            if hasFeatureMask is not None:
                ActualGrad = (
                    self.Grad.attribute(
                        input,
                        baselines=hasBaseline,
                        target=TestingLabel,
                        feature_mask=hasFeatureMask,
                    )
                    .data.cpu()
                    .numpy()
                )
            elif hasSliding_window_shapes is not None:
                # print("HAS SLIDING WINDOW")
                ActualGrad = (
                    self.Grad.attribute(
                        input,
                        sliding_window_shapes=hasSliding_window_shapes,
                        baselines=hasBaseline,
                        target=TestingLabel,
                    )
                    .data.cpu()
                    .numpy()
                )
            else:
                ActualGrad = (
                    self.Grad.attribute(
                        input, baselines=hasBaseline, target=TestingLabel
                    )
                    .data.cpu()
                    .numpy()
                )
        for t in range(sequence_length):
            newInput = input.clone()
            newInput[:, :, t] = assignment

            if hasBaseline is None:
                timeGrad_perTime = (
                    self.Grad.attribute(newInput, target=TestingLabel)
                    .data.cpu()
                    .numpy()
                )
            else:
                if hasFeatureMask is not None:
                    timeGrad_perTime = (
                        self.Grad.attribute(
                            newInput,
                            baselines=hasBaseline,
                            target=TestingLabel,
                            feature_mask=hasFeatureMask,
                        )
                        .data.cpu()
                        .numpy()
                    )
                elif hasSliding_window_shapes is not None:
                    # print("HAS SLIDING WINDOW")
                    timeGrad_perTime = (
                        self.Grad.attribute(
                            newInput,
                            sliding_window_shapes=hasSliding_window_shapes,
                            baselines=hasBaseline,
                            target=TestingLabel,
                        )
                        .data.cpu()
                        .numpy()
                    )
                else:
                    timeGrad_perTime = (
                        self.Grad.attribute(
                            newInput, baselines=hasBaseline, target=TestingLabel
                        )
                        .data.cpu()
                        .numpy()
                    )

            timeGrad_perTime = np.absolute(ActualGrad - timeGrad_perTime)
            timeGrad[:, t] = np.sum(timeGrad_perTime)

        timeContibution = preprocessing.minmax_scale(timeGrad, axis=1)
        meanTime = np.quantile(timeContibution, 0.55)

        for t in range(sequence_length):
            if timeContibution[0, t] > meanTime:
                for c in range(input_size):
                    newInput = input.clone()
                    newInput[:, c, t] = assignment

                    if hasBaseline is None:
                        inputGrad_perInput = (
                            self.Grad.attribute(newInput, target=TestingLabel)
                            .data.cpu()
                            .numpy()
                        )
                    else:
                        if hasFeatureMask is not None:
                            inputGrad_perInput = (
                                self.Grad.attribute(
                                    newInput,
                                    baselines=hasBaseline,
                                    target=TestingLabel,
                                    feature_mask=hasFeatureMask,
                                )
                                .data.cpu()
                                .numpy()
                            )
                        elif hasSliding_window_shapes is not None:
                            inputGrad_perInput = (
                                self.Grad.attribute(
                                    newInput,
                                    sliding_window_shapes=hasSliding_window_shapes,
                                    baselines=hasBaseline,
                                    target=TestingLabel,
                                )
                                .data.cpu()
                                .numpy()
                            )
                        else:
                            inputGrad_perInput = (
                                self.Grad.attribute(
                                    newInput, baselines=hasBaseline, target=TestingLabel
                                )
                                .data.cpu()
                                .numpy()
                            )

                    inputGrad_perInput = np.absolute(ActualGrad - inputGrad_perInput)
                    inputGrad[c, :] = np.sum(inputGrad_perInput)
                featureContibution = preprocessing.minmax_scale(inputGrad, axis=0)
            else:
                featureContibution = np.ones((input_size, 1)) * 0.1

            for c in range(input_size):
                newGrad[c, t] = timeContibution[0, t] * featureContibution[c, 0]
        # print('NewGrad',newGrad.shape)
        return newGrad

    def _givenAttGetRescaledSaliency(self, attributions, isTensor=True):
        if isTensor:
            saliency = np.absolute(attributions.data.cpu().numpy())
        else:
            saliency = np.absolute(attributions)
        saliency = saliency.reshape(-1, self.NumTimeSteps * self.NumFeatures)
        rescaledSaliency = preprocessing.minmax_scale(saliency, axis=1)
        rescaledSaliency = rescaledSaliency.reshape(attributions.shape)
        return rescaledSaliency
