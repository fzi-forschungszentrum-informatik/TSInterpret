from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

from TSInterpret.InterpretabilityModels.counterfactual.CF import CF
from TSInterpret.InterpretabilityModels.counterfactual.SETS.class_shapelets import (
    get_class_shapelets,
)
from TSInterpret.InterpretabilityModels.counterfactual.SETS.sets import sets_explain
from TSInterpret.InterpretabilityModels.counterfactual.SETS.shapelets import (
    ContractedShapeletTransform,
)
from TSInterpret.InterpretabilityModels.counterfactual.SETS.sktime_convert import (
    from_3d_numpy_to_nested,
)
from TSInterpret.InterpretabilityModels.counterfactual.SETS.utils import (
    MultivariateTransformer,
    get_indices,
    get_scores,
    get_shapelets,
    get_shapelets_distances,
)
from TSInterpret.Models.PyTorchModel import PyTorchModel
from TSInterpret.Models.SklearnModel import SklearnModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel


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
        time_contract_in_mins_per_dim,
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
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)
        test_y = le.transform(test_y)
        shape = test_x.shape
        if mode == "time":
            # Parse test data into (1, feat, time):
            change = True
            self.ts_length = shape[-2]
            train_x = np.swapaxes(train_x, 2, 1)
            test_x = np.swapaxes(
                test_x, 2, 1
            )  # test_x.reshape(test_x.shape[0], test_x.shape[2], test_x.shape[1])
        elif mode == "feat":
            change = False
            self.ts_length = shape[-1]

        self.data = [train_x, train_y, test_x, test_y]
        self.ts_len = train_x.shape[2]
        train_x_n = from_3d_numpy_to_nested(train_x)
        test_x_n = from_3d_numpy_to_nested(test_x)

        if backend == "PYT":
            self.predict = PyTorchModel(model, change).predict
        elif backend == "TF":
            self.predict = TensorFlowModel(model, change).predict

        elif backend == "SK":
            self.predict = SklearnModel(model, change).predict

        self.referenceset = (test_x, test_y)

        self.min_shapelet_len = 3
        self.max_shapelet_len = 20
        self.time_contract_in_mins_per_dim = time_contract_in_mins_per_dim
        self.initial_num_shapelets_per_case = 10

        self.trf_data = [train_x_n, train_y, test_x_n, test_y]

        # Required Shape (N,D,L)

        shapelet_transform = ContractedShapeletTransform(
            time_contract_in_mins=self.time_contract_in_mins_per_dim,
            num_candidates_to_sample_per_case=self.initial_num_shapelets_per_case,
            min_shapelet_length=self.min_shapelet_len,
            max_shapelet_length=self.max_shapelet_len,
            verbose=2,
            predefined_ig_rejection_level=0.001,
            max_shapelets_to_store_per_class=30,
        )
        # Fit multivaraite transformer
        transformer = MultivariateTransformer(shapelet_transform)
        transformer.fit(train_x_n, train_y)
        self.train_x_new = transformer.transform(train_x_n)
        # Save shapelets, scores, indices, distances for train
        self.transformer = transformer
        self.train_distances = get_shapelets_distances(self.transformer)
        self.shapelets = get_shapelets(self.transformer)
        self.scores = get_scores(self.transformer)
        self.indicies = get_indices(self.transformer)
        # Save distances for test
        self.test_x_new = transformer.transform(test_x_n)
        self.test_distances = get_shapelets_distances(transformer)
        (
            self.all_heat_maps,
            self.all_shapelets_class,
            self.all_shapelet_locations,
            self.all_shapelet_locations_test,
        ) = get_class_shapelets(
            self.trf_data,
            self.ts_len,
            self.shapelets,
            self.train_distances,
            self.test_distances,
        )

    def explain(
        self, ts_instance, x: np.ndarray, orig_class: int = None, target: int = None
    ) -> Tuple[np.ndarray, int]:
        """
        Calculates the Counterfactual according to Ates.
        Arguments:
            x (np.array): The instance to explain. Shape : `mode = time` -> `(1,time, feat)` or `mode = time` -> `(1,feat, time)`
            target int: target class. If no target class is given, the class with the secon heighest classification probability is selected.

        Returns:
            ([np.array], int): Tuple of Counterfactual and Label. Shape of CF : `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`

        """
        exp, label = sets_explain(
            ts_instance,
            self.model,
            self.data,
            self.ts_len,
            self.shapelets,
            self.all_shapelet_locations,
            self.all_shapelet_locations_test,
            self.all_shapelets_class,
            self.all_heat_maps,
            self.scores,
        )

        org_shape = x.shape
        return exp.reshape(org_shape), label
