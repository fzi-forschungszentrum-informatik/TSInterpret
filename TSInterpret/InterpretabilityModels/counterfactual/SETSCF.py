from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested

from TSInterpret.InterpretabilityModels.counterfactual.CF import CF
from TSInterpret.InterpretabilityModels.counterfactual.SETS.ContractedST import (
    ContractedShapeletTransform,
)
from TSInterpret.InterpretabilityModels.counterfactual.SETS.sets import (
    fit_shapelets,
    sets_explain,
)
from TSInterpret.InterpretabilityModels.counterfactual.SETS.utils import (
    MultivariateTransformer,
    get_scores,
    get_shapelets,
    get_shapelets_distances,
)
from TSInterpret.Models.PyTorchModel import PyTorchModel
from TSInterpret.Models.SklearnModel import SklearnModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel


class SETSCF(CF):
    """Calculates and Visualizes Counterfactuals for Multivariate Time Series in accordance to the paper [1].
        The shapelet transofor adapted by [1] is based on prior work of  [2].

    References
    ----------
    [1] Bahri, Omar, et al.
    "Shapelet-based Temporal Association Rule Mining for Multivariate Time Series Classification". SIGKDD 2022 Workshop on Mining and Learning from Time Series (MiLeTS)"
    [2] ostrom, Aaron and Bagnall, Anthony},
    "Binary shapelet transform for multiclass time series". Bostrom, Aaron and Bagnall, Anthony
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
        num_candidates_to_sample_per_case=20,
        time_contract_in_mins_per_dim=10,
        predefined_ig_rejection_level=0.001,
        max_shapelets_to_store_per_class=30,
        random_state=42,
        remove_self_similar=True,
        silent=False,
        fit_shapelets=True,
    ) -> None:
        """
        Arguments:
            model [torch.nn.Module, Callable, tf.keras.model]: Model to be interpreted.
            dataset: Reference Dataset of training and test data.
            backend str: desired Model Backend ('PYT', 'TF', 'SK').
            mode str: Name of second dimension: `time` -> `(-1, time, feature)` or `feat` -> `(-1, feature, time)`
            min_shapelet_len int: Value for min length of extracted shapelets / must be greater than 0
            max_shapelet_len int: Value for max length of extracted shapelets < timeseries must be smaller or equal than timeseries length
            num_candidates_to_sample_per_case int: number of assesed candiates per series
            time_contract_in_mins_per_dim int: Max time for shapelet extraction per dimension
            predefined_ig_rejection_levl float: Min Information Gain of candidate shapelet to keep
            random_state int: RandomState used throughout the shapelet transform
            remove_self_similar boolean: removes similar shapelets from a timeseries
            initial_num_shapelets_per_case int: Initial number of shapelets per case.
            silent bool: logging.
        """
        super().__init__(model, mode)
        self.backend = backend
        self.random_state = random_state
        # Parameters Shapelet Transform
        self.min_shapelet_len = min_shapelet_len
        self.max_shapelet_len = max_shapelet_len
        self.time_contract_in_mins_per_dim = time_contract_in_mins_per_dim
        self.initial_num_shapelets_per_case = num_candidates_to_sample_per_case
        # Prepare Data
        train_x, train_y = data
        self.le = LabelEncoder()
        self.train_y = self.le.fit_transform(train_y)
        if mode == "time":
            # Parse test data into (1, feat, time):
            change = True
            self.train_x = np.swapaxes(train_x, 2, 1)
            self.ts_len = train_x.shape[1]
        elif mode == "feat":
            change = False
            self.ts_len = train_x.shape[2]
        self.train_x_n = from_3d_numpy_to_nested(self.train_x)
        if backend == "PYT":
            self.predict = PyTorchModel(model, change).predict
        elif backend == "TF":
            self.predict = TensorFlowModel(model, change).predict
        elif backend == "SK":
            self.predict = SklearnModel(model, change).predict
        # Fit Shapelet Transform
        # Required Shape (N,D,L)
        if not silent:
            print(
                f"Extract Shapelets with information gain rejection lvl {predefined_ig_rejection_level} and shapelets per class of {max_shapelets_to_store_per_class}"
            )
        shapelet_transform = ContractedShapeletTransform(
            time_contract_in_mins=self.time_contract_in_mins_per_dim,
            num_candidates_to_sample_per_case=self.initial_num_shapelets_per_case,
            min_shapelet_length=self.min_shapelet_len,
            max_shapelet_length=self.max_shapelet_len,
            verbose=silent,
            predefined_ig_rejection_level=0.001,
            max_shapelets_to_store_per_class=30,
            remove_self_similar=remove_self_similar,
            random_state=self.random_state,
        )
        # Fit multivaraite transformer
        st_transformer = MultivariateTransformer(shapelet_transform)
        st_transformer.fit(self.train_x_n, train_y)
        self.train_x_new = st_transformer.transform(self.train_x_n)
        # Get Background Shapelet Distribution for Explainer
        self.st_transformer = st_transformer
        self.train_distances = get_shapelets_distances(self.st_transformer)
        self.shapelets = get_shapelets(self.st_transformer)
        self.scores = get_scores(self.st_transformer)
        # Get shapelet distances from transformer
        self.fitted_shapelets = (None,)
        self.threshhold = (None,)
        self.all_heat_maps = (None,)
        self.all_shapelets_class = (None,)

        if fit_shapelets == True:
            self.fit()

    def fit(self, occlusion_threshhold=1e-1, remove_multiclass_shapelets=True):
        """
        Calculates the occurences of shapelets and removes shapelets belonging to more than one class. This process can be triggered with different parameter options without a new shapelet transform run.
        Arguments:
            x (np.array): The instance to explain. Shape : `mode = time` -> `(1,time, feat)` or `mode = time` -> `(1,feat, time)`
            target int: target class. If no target class is given, the class with the secon heighest classification probability is selected.

        Returns:
            ([np.array], int): Tuple of Counterfactual and Label. Shape of CF : `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`

        """
        print(
            f"Fit function to prune shapelets with occlusion threshhold of {occlusion_threshhold} and remove shapelets belonging to more than one class set to {remove_multiclass_shapelets}"
        )
        (
            self.fitted_shapelets,
            self.threshhold,
            self.all_heat_maps,
            self.all_shapelets_class,
        ) = fit_shapelets(
            (self.train_x_n, self.train_y),
            self.ts_len,
            self.shapelets,
            self.train_distances,
            self.random_state,
            occlusion_threshhold,
            remove_multiclass_shapelets,
        )

    def explain(
        self, x: np.ndarray, orig_class: int = None, target: int = None
    ):  # -> Tuple[np.ndarray, int]:
        # x (np.array): The instance to explain. Shape : `mode = time` -> `(1,time, feat)` or `mode = time` -> `(1,feat, time)`
        # target int: target class. If no target class is given, the class with the secon heighest classification probability is selected.
        """
        Calculates the Counterfactual according to Ates.
        Arguments:
            x (np.array): The instance to explain. Shape : `mode = time` -> `(1,time, feat)` or `mode = time` -> `(1,feat, time)`
            target int: target class. If no target class is given, the class with the secon heighest classification probability is selected.

        Returns:
            ([np.array], int): Tuple of Counterfactual and Label. Shape of CF : `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`

        """
        if self.fitted_shapelets == None:
            print("Please use fit function first!")

        if target == None:
            target = list(np.unique(self.train_y))
        else:
            target = [target]

        expl, label = sets_explain(
            x,
            target,
            (self.train_x, self.train_y),
            self.st_transformer,
            self.model,
            self.ts_len,
            self.fitted_shapelets,
            self.threshhold,
            self.all_shapelets_class,
            self.all_heat_maps,
            self.scores,
            random_seed=self.random_state,
        )
        if expl is not None:
            org_shape = x.shape
            expl.reshape(org_shape)
            print("Counterfactual has been found")
        if expl is None:
            print("Could not find a cf for this timeseries")
        return expl, label
