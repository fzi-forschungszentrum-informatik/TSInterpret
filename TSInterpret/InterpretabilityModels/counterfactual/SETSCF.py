from typing import Tuple
import numpy as np
from TSInterpret.InterpretabilityModels.counterfactual.CF import CF
from TSInterpret.Models.PyTorchModel import PyTorchModel
from TSInterpret.Models.SklearnModel import SklearnModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel
from TSInterpret.InterpretabilityModels.counterfactual.SETS.shapelets import ContractedShapeletTransform
from TSInterpret.InterpretabilityModels.counterfactual.SETS.utils import MultivariateTransformer, get_shapelets_distances

class SETSCF(CF):
    """Calculates and Visualizes Counterfactuals for Multivariate Time Series in accordance to the paper [1].

    References
    ----------
     [1] Ates, Emre, et al.
     "Shapelet-based Temporal Association Rule Mining for Multivariate Time Series Classification"
     2021 International Conference on Applied Artificial Intelligence (ICAPAI). IEEE, 2021.
    ----------
    """

    def __init__(
        self,
        model,
        data,
        backend,
        mode, 
        method="opt",
        number_distractors=2,
        max_attempts=1000,
        max_iter=1000,
        silent=False,
    ) -> None:
        """
        Arguments:
            model [torch.nn.Module, Callable, tf.keras.model]: Model to be interpreted.
            ref Tuple: Reference Dataset as Tuple (x,y).
            backend str: desired Model Backend ('PYT', 'TF', 'SK').
            mode str: Name of second dimension: `time` -> `(-1, time, feature)` or `feat` -> `(-1, feature, time)`
            method str : 'opt' if optimized calculation, 'brute' for Brute Force
            number_distractors int: number of distractore to be used
            silent bool: logging.

        """
        super().__init__(model, mode)
        self.backend = backend
        train_x, train_y, test_x, test_y = data
        shape = test_x.shape
        if mode == "time":
            # Parse test data into (1, feat, time):
            change = True
            self.ts_length = shape[-2]
            test_x = np.swapaxes(
                test_x, 2, 1
            )  # test_x.reshape(test_x.shape[0], test_x.shape[2], test_x.shape[1])
        elif mode == "feat":
            change = False
            self.ts_length = shape[-1]

        if backend == "PYT":
            self.predict = PyTorchModel(model, change).predict
        elif backend == "TF":
            self.predict = TensorFlowModel(model, change).predict

        elif backend == "SK":
            self.predict = SklearnModel(model, change).predict

        self.referenceset = (test_x, test_y)

        self.min_shapelet_len = 3
        self.max_shapelet_len = 20
        self.time_contract_in_mins_per_dim = 1
        self.initial_num_shapelets_per_case = 10

        # Required Shape (N,D,L)

        shapelet_transform = ContractedShapeletTransform(
            time_contract_in_mins=self.time_contract_in_mins_per_dim,
            num_candidates_to_sample_per_case=self.initial_num_shapelets_per_case,
            min_shapelet_length=self.min_shapelet_len,
            max_shapelet_length=self.max_shapelet_len,
            verbose=0,
            predefined_ig_rejection_level=0.05,
            max_shapelets_to_store_per_class=30,
        )

        transformer = MultivariateTransformer(shapelet_transform)
        transformer.fit(train_x, train_y)
        train_x_new = transformer.transform(train_x)
        test_x_new  = transformer.transform(test_x)
        self.train_distances = get_shapelets_distances(train_x_new)
        self.test_distances  = get_shapelets_distances(test_x_new)




    def explain(
        self, x: np.ndarray, orig_class: int = None, target: int = None
    ) -> Tuple[np.ndarray, int]:
        """
        Calculates the Counterfactual according to Ates.
        Arguments:
            x (np.array): The instance to explain. Shape : `mode = time` -> `(1,time, feat)` or `mode = time` -> `(1,feat, time)`
            target int: target class. If no target class is given, the class with the secon heighest classification probability is selected.

        Returns:
            ([np.array], int): Tuple of Counterfactual and Label. Shape of CF : `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`

        """
        org_shape = x.shape
        if self.mode != "feat":
            x = np.swapaxes(x, -1, -2)  # x.reshape(-1, x.shape[-1], x.shape[-2])
        train_x, train_y = self.referenceset
        if len(train_y.shape) > 1:
            train_y = np.argmax(train_y, axis=1)
        if self.method == "opt":
            opt = OptimizedSearch(
                self.predict,
                train_x,
                train_y,
                silent=self.silent,
                threads=1,
                num_distractors=self.number_distractors,
                max_attempts=self.max_attemps,
                maxiter=self.max_iter,
            )
            exp, label = opt.explain(x, to_maximize=target)
        elif self.method == "brute":
            opt = BruteForceSearch(self.predict, train_x, train_y, threads=1)
            exp, label = opt.explain(x, to_maximize=target)
        if self.mode != "feat":
            exp = np.swapaxes(exp, -1, -2)
        return exp.reshape(org_shape), label
