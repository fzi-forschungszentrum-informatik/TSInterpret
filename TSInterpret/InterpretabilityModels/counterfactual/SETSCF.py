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
     [1] Bahri, Omar, et al.
    "Shapelet-based Temporal Association Rule Mining for Multivariate Time Series Classification". SIGKDD 2022 Workshop on Mining and Learning from Time Series (MiLeTS)
    ----------
    """

    def __init__(
        self,
        model,
        data,
        backend,
        mode,
        min_shapelet_len,
        max_shapelet_len,
        time_contract_in_mins_per_dim,
        initial_num_shapelets_per_case,
        silent=False,
    ) -> None:
        """
        Arguments:
            model [torch.nn.Module, Callable, tf.keras.model]: Model to be interpreted.
            dataset: Reference Dataset of training and test data.
            backend str: desired Model Backend ('PYT', 'TF', 'SK').
            mode str: Name of second dimension: `time` -> `(-1, time, feature)` or `feat` -> `(-1, feature, time)`
            min_shapelet_len int: Value for min length of extracted shapelets / must be greater than 0
            max_shapelet_len int: Value for max length of extracted shapelets < timeseries must be smaller or equal than timeseries length
            time_contract_in_mins_per_dim int: Max time for shapelet extraction per dimension
            initial_num_shapelets_per_case int: Initial number of shapelets per case.
            silent bool: logging.

        """
        super().__init__(model, mode)
        self.backend = backend
        # Parameters Shapelet Transform
        self.min_shapelet_len = min_shapelet_len
        self.max_shapelet_len = max_shapelet_len
        self.time_contract_in_mins_per_dim = time_contract_in_mins_per_dim
        self.initial_num_shapelets_per_case = initial_num_shapelets_per_case
        # Prepare Data
        train_x, train_y, test_x, test_y = data
        self.le = LabelEncoder()
        train_y = self.le.fit_transform(train_y)
        test_y = self.le.transform(test_y)
        shape = test_x.shape
        if mode == "time":
            # Parse test data into (1, feat, time):
            change = True
            train_x = np.swapaxes(train_x, 2, 1)
            test_x = np.swapaxes(test_x, 2, 1)
        elif mode == "feat":
            change = False
            self.ts_length = shape[-1]
        self.ts_len = train_x.shape[2]
        self.train_x_n = from_3d_numpy_to_nested(train_x)
        self.test_x_n = from_3d_numpy_to_nested(test_x)
        self.data = [train_x, train_y, test_x, test_y]
        self.trf_data = [self.train_x_n, train_y, self.test_x_n, test_y]
        # Prepare models
        if backend == "PYT":
            self.predict = PyTorchModel(model, change).predict
        elif backend == "TF":
            self.predict = TensorFlowModel(model, change).predict
        elif backend == "SK":
            self.predict = SklearnModel(model, change).predict
        # Fit Shapelet Transform
        # Required Shape (N,D,L)
        shapelet_transform = ContractedShapeletTransform(
            time_contract_in_mins=self.time_contract_in_mins_per_dim,
            num_candidates_to_sample_per_case=self.initial_num_shapelets_per_case,
            min_shapelet_length=self.min_shapelet_len,
            max_shapelet_length=self.max_shapelet_len,
            verbose=silent,
            predefined_ig_rejection_level=0.001,
            max_shapelets_to_store_per_class=30,
        )
        # Fit multivaraite transformer
        transformer = MultivariateTransformer(shapelet_transform)
        transformer.fit(self.train_x_n, train_y)
        self.train_x_new = transformer.transform(self.train_x_n)
        # Save shapelets, scores, indices, distances for train
        self.transformer = transformer
        self.train_distances = get_shapelets_distances(self.transformer)
        self.shapelets = get_shapelets(self.transformer)
        self.scores = get_scores(self.transformer)
        self.indicies = get_indices(self.transformer)
        # Save distances for test
        self.test_x_new = transformer.transform(self.test_x_n)
        # Get shapelet distances from transformer
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
    ):  # -> Tuple[np.ndarray, int]:
        # x (np.array): The instance to explain. Shape : `mode = time` -> `(1,time, feat)` or `mode = time` -> `(1,feat, time)`
        # target int: target class. If no target class is given, the class with the secon heighest classification probability is selected.
        """
        Calculates the Counterfactual according to Ates.
        Arguments:
            int: Index of timeseries to generate a counterfactual explanation
        Returns:
            ([np.array], int): Tuple of Counterfactual and Label. Shape of CF : `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`

        """
        print(self.all_shapelets_class.values())
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
        # if exp is not None:
        #    org_shape = x.shape
        #    exp.reshape(org_shape)
        return exp, label
