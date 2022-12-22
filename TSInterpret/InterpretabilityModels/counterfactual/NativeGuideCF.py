"""Implementation after Delaney et al . https://github.com/e-delaney/Instance-Based_CFE_TSC"""
import warnings
from typing import Tuple

import numpy as np
from torchcam.methods import CAM
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.neighbors import KNeighborsTimeSeries

from TSInterpret.InterpretabilityModels.counterfactual.CF import CF
from TSInterpret.InterpretabilityModels.GradCam.GradCam_1D import GradCam1D
from TSInterpret.Models.PyTorchModel import PyTorchModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


class NativeGuideCF(CF):
    """
    NUN_CF according to [1] for both torch and tensorflow.

    References
    ----------
    [1] Delaney, E., Greene, D., Keane, M.T.: Instance-Based Counterfactual Explanations
        for Time Series Classification. In: Sanchez-Ruiz, A.A., Floyd, M.W. (eds.) Case-
        Based Reasoning Research and Development, vol. 12877, pp. 32–47. Springer
        International Publishing, Cham (2021), series Title: Lecture Notes in Computer
        Science.
    ----------
    """

    def __init__(
        self,
        model,
        shape,
        reference_set,
        backend="PYT",
        mode="feat",
        method="NUN-CF",
        distance_measure="dtw",
        n_neighbors=1,
        max_iter=500,
    ) -> None:
        """
        In this case differentiation between time & feat not necessary as implicitly given by CNN. Only works for CNNs due to the attribution methods.
        Arguments:
            model [torch.nn.Module, Callable, tf.keras.model]: classification model to explain
            shape Tuple: input shape
            reference_set Tuple: reference set as tuple (x,y)
            backend str: 'PYT' or  'TF'
            mode str: model either 'time' or 'feat'. `time` -> `(-1, time, feature)` or `feat` -> `(-1, feature, time)`
            method str: 'Nun_CF', 'dtw_bary_center' or 'native_guide'.
            distance_measure str: sklearn appreviation for distance of knn.
            n_neighbore int: # neighbors to select from.
            max_iter int : max number of runs
        """
        super().__init__(model, mode)
        self.backend = backend
        test_x, test_y = reference_set
        test_x = np.array(test_x, dtype=np.float32)

        if mode == "time":
            # Parse test data into (1, feat, time):
            self.ts_length = shape[-2]
            test_x = test_x.reshape(test_x.shape[0], test_x.shape[2], test_x.shape[1])
        elif mode == "feat":
            self.ts_length = shape[-1]

        if backend == "PYT":

            try:
                self.cam_extractor = CAM(self.model, input_shape=(shape[1], shape[2]))
            except:
                print("GradCam Hook already registered")
            change = False
            if self.mode == "time":
                change = True
            self.predict = PyTorchModel(self.model, change).predict
            y_pred = np.argmax(self.predict(test_x), axis=1)

        elif backend == "TF":
            self.cam_extractor = GradCam1D()  # VanillaGradients()#GradCam1D()
            y_pred = np.argmax(
                self.model.predict(test_x.reshape(-1, self.ts_length, 1)), axis=1
            )
            self.predict = TensorFlowModel(self.model, change=True).predict
        else:
            print("Only Compatible with Tensorflow (TF) or Pytorch (PYT)!")

        self.reference_set = (test_x, y_pred)
        self.method = method
        self.distance_measure = distance_measure
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        # Manipulate reference set replace original y with predicted y

    def _native_guide_retrieval(self, query, predicted_label, distance, n_neighbors):
        """
        This gets the nearest unlike neighbors.
        Arguments:
            query (np.array): The instance to explain.
            predicted_label (np.array): Label of instance.
            reference_set (np.array): Set of addtional labeled data (could be training or test set)
            distance (str):
            n_neighbors (int):number nearest neighbors to return
        Returns:
            [np.array]: Returns K_Nearest_Neighbors of input query with different classification label.

        """
        if type(predicted_label) != int:
            predicted_label = np.argmax(predicted_label)

        x_train, y = self.reference_set
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        ts_length = self.ts_length
        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
        knn.fit(x_train[list(np.where(y != predicted_label))].reshape(-1, ts_length, 1))
        dist, ind = knn.kneighbors(query.reshape(1, ts_length, 1), return_distance=True)
        x_train.reshape(-1, 1, ts_length)
        return dist[0], x_train[np.where(y != predicted_label)][ind[0]]

    def _native_guide_wrapper(self, query, predicted_label, distance, n_neighbors):
        _, nun = self._native_guide_retrieval(
            query, predicted_label, distance, n_neighbors
        )
        individual = np.array(nun.tolist(), dtype=np.float64)
        out = self.predict(individual)
        return nun, np.argmax(out)

    def _findSubarray(
        self, a, k
    ):  # used to find the maximum contigious subarray of length k in the explanation weight vector

        n = len(a)

        vec = []

        # Iterate to find all the sub-arrays
        for i in range(n - k + 1):
            temp = []

            # Store the sub-array elements in the array
            for j in range(i, i + k):
                temp.append(a[j])

            # Push the vector in the container
            vec.append(temp)

        sum_arr = []
        for v in vec:
            sum_arr.append(np.sum(v))

        return vec[np.argmax(sum_arr)]

    def _counterfactual_generator_swap(
        self, instance, label, subarray_length=1, max_iter=500
    ):
        print(label)
        _, nun = self._native_guide_retrieval(instance, label, self.distance_measure, 1)
        if np.count_nonzero(nun.reshape(-1) - instance.reshape(-1)) == 0:
            print("Starting and nun are Identical !")

        test_x, test_y = self.reference_set
        train_x = test_x
        individual = np.array(nun.tolist(), dtype=np.float64)
        out = self.predict(individual)
        if self.backend == "PYT":
            training_weights = (
                self.cam_extractor(out.squeeze(0).argmax().item(), out)[0]
                .detach()
                .numpy()
            )
        elif self.backend == "TF":
            data = (instance.reshape(1, -1, 1), None)
            training_weights = self.cam_extractor.explain(
                data, self.model, class_index=label[0]
            )  # grad_cam(self.model, instance.reshape(1,-1,1))#self.cam_extractor.explain(data, self.model,class_index=label)#instance
        # Classify Original
        individual = np.array(instance.tolist(), dtype=np.float64)
        out = self.predict(individual)

        most_influencial_array = self._findSubarray((training_weights), subarray_length)

        starting_point = np.where(training_weights == most_influencial_array[0])[0][0]

        X_example = instance.copy().reshape(1, -1)

        nun = nun.reshape(1, -1)
        X_example[0, starting_point : subarray_length + starting_point] = nun[
            0, starting_point : subarray_length + starting_point
        ]
        individual = np.array(
            X_example.reshape(-1, 1, train_x.shape[-1]).tolist(), dtype=np.float64
        )
        out = self.predict(individual)
        prob_target = out[0][
            label
        ]  # torch.nn.functional.softmax(model(torch.from_numpy(test_x))).detach().numpy()[0][y_pred[instance]]
        counter = 0
        while prob_target > 0.5 and counter < max_iter:

            subarray_length += 1
            most_influencial_array = self._findSubarray(
                (training_weights), subarray_length
            )
            starting_point = np.where(training_weights == most_influencial_array[0])[0][
                0
            ]
            X_example = instance.copy().reshape(1, -1)
            X_example[:, starting_point : subarray_length + starting_point] = nun[
                :, starting_point : subarray_length + starting_point
            ]
            individual = np.array(
                X_example.reshape(-1, 1, train_x.shape[-1]).tolist(), dtype=np.float64
            )
            out = self.predict(individual)
            prob_target = out[0][label]
            counter = counter + 1
            if counter == max_iter or subarray_length == self.ts_length:
                print("No Counterfactual found")
                return None, None

        return X_example, np.argmax(out, axis=1)[0]

    def _instance_based_cf(self, query, label, target, distance="dtw", max_iter=500):

        d, nan = self._native_guide_retrieval(query, label, distance, 1)
        beta = 0
        insample_cf = nan.reshape(1, 1, -1)

        individual = np.array(query.tolist(), dtype=np.float64)

        output = self.predict(individual)
        pred_treshold = 0.5
        target = np.argsort(output)[0][-2:-1][0]
        query = query.reshape(-1)
        insample_cf = insample_cf.reshape(-1)
        generated_cf = dtw_barycenter_averaging(
            [query, insample_cf], weights=np.array([(1 - beta), beta])
        )
        generated_cf = generated_cf.reshape(1, 1, -1)
        individual = np.array(generated_cf.tolist(), dtype=np.float64)
        prob_target = self.predict(individual)[0][target]
        counter = 0

        while prob_target < pred_treshold and counter < max_iter:
            beta += 0.01
            generated_cf = dtw_barycenter_averaging(
                [query, insample_cf], weights=np.array([(1 - beta), beta])
            )
            generated_cf = generated_cf.reshape(1, 1, -1)
            individual = np.array(generated_cf.tolist(), dtype=np.float64)
            prob_target = self.predict(individual)[0][target]

            counter = counter + 1
        if counter == max_iter:
            print("No Counterfactual found")
            return None, None

        return generated_cf, target

    def explain(self, x: np.ndarray, y: int) -> Tuple[np.array, int]:
        """'
        Explains a specific instance x.
        Arguments:
            x np.array : instance to be explained.
            y int: predicted label for instance x.
        Returns:
            Tuple: (counterfactual, counterfactual label)

        """
        if self.mode == "time":
            x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
            print(x.shape)
        if self.method == "NG":
            return self._native_guide_wrapper(
                x, y, self.distance_measure, self.n_neighbors
            )
        elif self.method == "dtw_bary_center":
            return self._instance_based_cf(x, y, self.distance_measure)
        elif self.method == "NUN_CF":
            self.distance_measure = "euclidean"
            return self._counterfactual_generator_swap(x, y, max_iter=self.max_iter)
        else:
            print("Unknown Method selected.")
