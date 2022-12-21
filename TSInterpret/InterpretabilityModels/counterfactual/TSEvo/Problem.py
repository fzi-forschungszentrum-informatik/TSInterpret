import numpy as np
import torch
from pymop import Problem


class MultiObjectiveCounterfactuals(Problem):
    """
    Multiobjective Problem for the Evaluation of TSEvo.

    """

    def __init__(
        self,
        model,
        observation,
        original_y,
        target,
        reference_set,
        neighborhood,
        window,
        backend="torch",
        channels=1,
    ):

        super().__init__(n_var=channels, n_obj=3, n_constr=0, evaluation_of="auto")

        self.model = model

        self.window = window
        self.observation = observation
        print(self.observation.shape)
        self.target = target
        self.original_y = original_y
        if len(original_y) > 1:
            self.original_label = np.argmax(original_y, axis=1)[0]
        else:
            self.original_label = original_y

        self.reference_set = reference_set
        # self.max= max(reference_set)
        self.neighborhood = neighborhood
        self.backend = backend

        if self.backend == "PYT":
            self.model.eval()
        # print(type(original_y))
        if type(original_y) == np.int64:  # or original_y.shape[1]==2 :
            print("Binary Case")
            if self.backend == "PYT":
                print("Predict Torch")
                self.predict = self.get_prediction_torch
            if self.backend == "TF":
                print("Predict TF")
                self.predict = self.get_prediction_tensorflow
            self.output_distance = self.output_distance_binary
        elif self.target is None:
            print("No Target")
            if self.backend == "PYT":
                self.predict = self.get_prediction_torch
            if self.backend == "TF":
                self.predict = self.get_prediction_tensorflow
            self.output_distance = self.output_distance_multi
        else:
            print("Target")
            if self.backend == "PYT":
                self.predict = self.get_prediction_torch
            if self.backend == "TF":
                self.predict = self.get_prediction_tensorflow
            self.output_distance = self.output_distance_target
        self.label, self.output = self.predict(observation, full=True)

    def _evaluate(self, explanation, out, *args, **kwargs):

        label, output = self.predict(explanation)

        output_distance = self.output_distance(output, label)

        x_distance = self.mean_delta(self.observation, explanation)

        num_changed_features = np.count_nonzero(self.observation - explanation) / (
            self.observation.shape[-1] * self.observation.shape[-2]
        )  # TODO Used to be 150 !

        out["F"] = np.column_stack([output_distance, x_distance, num_changed_features])

    def feasible(self, individual):
        """Feasibility function for the individual. Returns True if feasible False
        otherwise."""
        # TODO
        prediction, _ = self.predict(np.asarray(individual))

        if prediction != self.original_label:
            return True

        return False

    # todo normalize
    def mean_delta(self, first, second):
        return np.sum(np.abs(first - second)) / (first.shape[-1] * first.shape[-2])

    def get_prediction_torch(
        self,
        individual,
        full=False,
    ):

        individual = np.array(individual.tolist(), dtype=np.float64)
        input_ = torch.from_numpy(individual).float().reshape(1, -1, self.window)

        with torch.no_grad():
            output = torch.nn.functional.softmax(self.model(input_)).detach().numpy()

        idx = output.argmax()

        if full:
            return idx, output[0]
        return idx, output[0][idx]

    def get_prediction_tensorflow(self, individual, full=False):
        individual = np.array(individual.tolist(), dtype=np.float64)
        output = self.model.predict(individual.reshape(1, self.window, -1), verbose=0)
        idx = output.argmax()

        if full:
            return idx, output[0]
        return idx, output[0][idx]

    def get_prediction_target_torch(self, individual, full=False, binary=False):
        individual = np.array(individual.tolist(), dtype=np.float64)
        input_ = torch.from_numpy(individual).float().reshape(1, -1, self.window)
        output = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        idx = output.argmax()
        if full:
            return idx, output[0]
        return idx, output[0][idx]

    def get_prediction_target_tensorflow(self, individual, full=False):
        individual = np.array(individual.tolist(), dtype=np.float64)
        output = self.model.predict(individual.reshape(1, self.window, -1), verbose=0)
        idx = output.argmax()
        if full:
            return idx, output[0]
        return idx, output[0][idx]

    def output_distance_binary(self, output, label):

        output_distance = 1 - output

        if label == self.original_label:

            output_distance = 1.0  # output

        return output_distance

    def output_distance_multi(self, output, label):

        output_distance = output - self.output[label]

        if label == self.original_label:
            output_distance = 1.0  # output
        return output_distance

    def output_distance_target(self, output, label):

        output_distance = output - self.output[self.target]
        if label != self.target:
            output_distance = 1.0  # output
        return output_distance
