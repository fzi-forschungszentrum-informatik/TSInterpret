import numpy as np
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

from TSInterpret.InterpretabilityModels.leftist.learning_process.learning_process import (
    LearningProcess,
)
from TSInterpret.InterpretabilityModels.leftist.learning_process.neighbors_generator.LIME_neighbors_generator import (
    LIMENeighborsGenerator,
)

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class LIMELearningProcess(LearningProcess):
    """
    Module to explain neighbors as in LIME.

    Attributes:
        random_state (np.random.RandomState): random state for the regressor.
        feature_selection (string): name of the feature selection.
        model_regressor (..): the explanation model type.
    """

    def __init__(
        self,
        random_state,
        feature_selection_method="auto",
        model_regressor=None,
        distance_metric=None,
    ):
        """
        Must inherit LearningProcess class.

        Parameters:
            random_state (np.random.RandomState): random state for the regressor.
            feature_selection (string): name of the feature selection.
                                        If 'auto' select one according the number of features.
            distance_metric (string): name of the dsitance metric for neighbors generation
            model_regressor (..): the explanation model type.
                                    If None, use Ridge model.
        """
        LearningProcess.__init__(self)
        self.random_state = random_state
        self.feature_selection_method = feature_selection_method
        self.explanation_size = None
        if model_regressor is None:
            self.model_regressor = Ridge(
                alpha=1, fit_intercept=True, random_state=self.random_state
            )
        random_state_generator = check_random_state(None)
        self.neighbors_generator = LIMENeighborsGenerator(
            random_state_generator, distance_metric=distance_metric
        )

    def solve(self, neighbors, idx_label, explanation_size=None):
        """
        Build the explanation model from the neighbors as in LIME.

        Parameters:
            neighbors (Neighbors): the neighbors.
            idx_label (int): index of the label to explain.
            explanation_size (int,None): size of the explanation (number of features to use in model explanation).
                                    If None explanation_size = nb_features.

        Returns:
            explanation (Explanation): the coefficients of the explanation model.
        """
        if explanation_size is None:
            _explanation_size = neighbors.masks.shape[1]
        elif isinstance(explanation_size, int):
            _explanation_size = explanation_size
        else:
            raise TypeError("explanation_size must an int or None")

        # get the proba classification from the desired label
        proba_label_column = neighbors.proba_labels[:, idx_label]

        # select the features to use for the explanation model
        used_features = self._feature_selection(
            neighbors, proba_label_column, _explanation_size
        )

        if used_features is None:
            used_features = list(range(neighbors.masks.shape[1]))

        # fit the explanation model on the neighbors
        easy_model = self.model_regressor

        easy_model.fit(
            neighbors.masks[:, used_features],
            proba_label_column,
            sample_weight=neighbors.kernel_weights,
        )

        # compute performance of explanation model on the neighbors
        prediction_score = easy_model.score(
            neighbors.masks[:, used_features],
            proba_label_column,
            sample_weight=neighbors.kernel_weights,
        )

        # proba predict by the explanation for the instance to explain
        explained_instance_proba_predict = easy_model.predict(
            neighbors.masks[0, used_features].reshape(1, -1)
        )

        coeff = sorted(zip(used_features, easy_model.coef_))

        if len(used_features) < neighbors.masks.shape[1]:
            coeff_null = np.array(
                [
                    tuple([el, 0.0])
                    for el in list(
                        set(range(neighbors.masks.shape[1])) - set(used_features)
                    )
                ]
            )
            coeff = np.concatenate((coeff, coeff_null))

        return (
            np.array([el[1] for el in np.sort(coeff, axis=0)]),
            easy_model.intercept_,
            prediction_score,
            explained_instance_proba_predict,
        )

    def _feature_selection(self, neighbors, proba_label_column, explanation_size):
        """
        Selects features for the model. see explain_instance_with_data to understand the parameters.

        Parameters:
            neighbors (Neighbors): the neighbors.
            proba_label_column (np.ndarray): proba classification of the neighbors for one label.
            explanation_size (int): size of the explanation (number of features to use in model explanation)

        Returns:
            used_features (np.ndarray): index of the features to use for building the explanation model.

        """
        if explanation_size > neighbors.masks.shape[1]:
            explanation_size = neighbors.masks.shape[1]

        if self.feature_selection_method == "none":
            return np.array(range(neighbors.masks.shape[1]))

        elif self.feature_selection_method == "forward_selection":
            return self._ft_selection_forward_selection(
                neighbors, proba_label_column, explanation_size
            )

        elif self.feature_selection_method == "highest_weights":
            return self._ft_selection_highest_weights(
                neighbors, proba_label_column, explanation_size
            )

        elif self.feature_selection_method == "lasso_path":
            return self._ft_selection_lasso_path(
                neighbors, proba_label_column, explanation_size
            )

        elif self.feature_selection_method == "auto":
            if explanation_size <= 6:
                self.feature_selection_method = "forward_selection"
            else:
                self.feature_selection_method = "highest_weights"
            return self._feature_selection(
                neighbors, proba_label_column, explanation_size
            )

    def _ft_selection_forward_selection(
        self, neighbors, proba_label_column, explanation_size
    ):
        """
        Feature selection by iteratively adding features to the model.

        Parameters:
            neighbors_masks (Neighors): the neighbors.
            proba_label_column (np.ndarray): proba classification of the neighbors for one label.
            explanation_size (int): size of the explanation (number of features to use in model explanation)

        Returns:
            used_features (np.ndarray): index of the features to use for building the explanation model.
        """

        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(explanation_size, neighbors.masks.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(neighbors.masks.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(
                    neighbors.masks[:, used_features + [feature]],
                    proba_label_column,
                    sample_weight=neighbors.kernel_weights,
                )
                score = clf.score(
                    neighbors.masks[:, used_features + [feature]],
                    proba_label_column,
                    sample_weight=neighbors.kernel_weights,
                )
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def _ft_selection_lasso_path(self, neighbors, proba_label_column, explanation_size):
        """
        Feature selection with lasso path.

        Parameters:
            neighbors (Neighbors): the neighbors.
            proba_label_column (np.ndarray): proba classification of the neighbors for one label.
            explanation_size (int): size of the explanation (number of features to use in model explanation)

        Returns:
            used_features (np.ndarray): index of the features to use for building the explanation model.
        """
        weighted_neighbors_masks = (
            neighbors.masks
            - np.average(neighbors.masks, axis=0, weights=neighbors.kernel_weights)
        ) * np.sqrt(neighbors.kernel_weights[:, np.newaxis])
        weighted_proba_label_column = (
            proba_label_column
            - np.average(proba_label_column, weights=neighbors.kernel_weights)
        ) * np.sqrt(neighbors.kernel_weights)
        nonzero = range(weighted_neighbors_masks.shape[1])
        x_vector = weighted_neighbors_masks
        alphas, _, coefs = lars_path(
            x_vector, weighted_proba_label_column, method="lasso", verbose=False
        )
        for i in range(len(coefs.T) - 1, 0, -1):
            nonzero = coefs.T[i].nonzero()[0]
            if len(nonzero) <= explanation_size:
                break
        return nonzero

    def _ft_selection_highest_weights(
        self, neighbors, proba_label_column, explanation_size
    ):
        """
        Feature selection with by highest weight.

        Parameters:
            neighbors (Neighbors): the neighbors.
            proba_label_column (np.ndarray): proba classification of the neighbors for one label.
            explanation_size (int): size of the explanation (number of features to use in model explanation)

        Returns:
            used_features (np.ndarray): index of the features to use for building the explanation model.
        """
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        clf.fit(
            neighbors.masks, proba_label_column, sample_weight=neighbors.kernel_weights
        )
        feature_weights = sorted(
            zip(range(neighbors.masks.shape[0]), clf.coef_ * neighbors.masks[0]),
            key=lambda x: np.abs(x[1]),
            reverse=True,
        )

        return np.array([x[0] for x in feature_weights[:explanation_size]])
