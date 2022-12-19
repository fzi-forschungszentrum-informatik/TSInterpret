import numpy as np
from sklearn.linear_model import lars_path

from TSInterpret.InterpretabilityModels.leftist.learning_process.learning_process import (
    LearningProcess,
)
from TSInterpret.InterpretabilityModels.leftist.learning_process.neighbors_generator.SHAP_neighbors_generator import (
    SHAPNeighborsGenerator,
)

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class SHAPLearningProcess(LearningProcess):
    def __init__(
        self,
        explained_instance,
        model_to_explain,
        external_dataset=None,
        init_shap_explainer=None,
    ):
        """
        Must inherit LearningProcess class.
        """
        LearningProcess.__init__(self)
        external_data = external_dataset
        self.l1_reg = "auto"
        if init_shap_explainer is None:
            self.mean_background_dataset_proba_labels = (
                self._classify_background_dataset(model_to_explain, external_data)
            )
        else:
            self.mean_background_dataset_proba_labels = init_shap_explainer
        self.explained_instance_proba_labels = self._classify_explained_instance(
            model_to_explain,
            explained_instance.reshape(
                1, explained_instance.shape[0], explained_instance.shape[1]
            ),
        )
        self.neighbors_generator = SHAPNeighborsGenerator()

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
        nb_features = neighbors.masks.shape[1]
        # nb_neighbors = neighbors.masks.shape[0]

        eyAdj = (
            neighbors.proba_labels[:, idx_label]
            - self.mean_background_dataset_proba_labels[idx_label]
        )
        s = np.sum(neighbors.masks, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(nb_features)
        if explanation_size is not None:
            w_aug = np.hstack(
                (
                    neighbors.kernel_weights * (nb_features - s),
                    neighbors.kernel_weights * s,
                )
            )
            w_sqrt_aug = np.sqrt(w_aug)

            eyAdj_aug = np.hstack(
                (
                    eyAdj,
                    eyAdj
                    - (
                        self.explained_instance_proba_labels[idx_label]
                        - self.mean_background_dataset_proba_labels[idx_label]
                    ),
                )
            )
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(
                w_sqrt_aug
                * np.transpose(np.vstack((neighbors.masks, neighbors.masks - 1)))
            )

            nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=explanation_size)[1]

        if len(nonzero_inds) == 0:
            return np.zeros(nb_features), np.ones(nb_features)

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - neighbors.masks[:, nonzero_inds[-1]] * (
            self.explained_instance_proba_labels[idx_label]
            - self.mean_background_dataset_proba_labels[idx_label]
        )
        etmp = np.transpose(
            np.transpose(neighbors.masks[:, nonzero_inds[:-1]])
            - neighbors.masks[:, nonzero_inds[-1]]
        )

        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(np.transpose(etmp) * np.transpose(neighbors.kernel_weights))
        tmp2 = np.linalg.inv(np.dot(np.transpose(tmp), etmp))
        w = np.dot(tmp2, np.dot(np.transpose(tmp), eyAdj2))
        phi = np.zeros(nb_features)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (
            self.explained_instance_proba_labels[idx_label]
            - self.mean_background_dataset_proba_labels[idx_label]
        ) - sum(w)

        # clean up any rounding errors
        for i in range(nb_features):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))

    def _classify_background_dataset(self, model_to_explain, background_dataset):
        """
        Classify the background dataset by the model to explain and mean the obtained proba labels.

        Returns:
            self.mean_background_dataset_proba_labels (np.ndarray): mean of the classification of the background dataset.
        """
        predictions = model_to_explain(background_dataset)
        if len(predictions[0]) == 1:
            predictions = np.array([np.array([el[0], 1 - el[0]]) for el in predictions])
        return np.mean(predictions, axis=0)

    def _classify_explained_instance(self, model_to_explain, explained_instance):
        """
        Classify the instance to explain by the model to explain.

        Returns:
            explained_instance_classification (np.ndarray): the classification of the instance to explain.
        """
        predictions = model_to_explain(explained_instance)[0]
        if len(predictions) == 1:
            predictions = np.array([predictions[0], 1 - predictions[0]])
        return predictions
