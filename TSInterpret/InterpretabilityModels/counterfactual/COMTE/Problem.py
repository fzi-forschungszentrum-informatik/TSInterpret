import numpy as np


class Problem:
    def __init__(self, length, loss, max_val):
        self.loss = loss
        self.max_val = max_val
        self.length = length
        # self.maximize=maximize
        self.fitness = np.inf
        self.state = np.array([0] * self.length)

    def random_neighbor(self):
        neighbor = np.copy(self.state)
        i = np.random.randint(0, self.length)

        if self.max_val == 2:
            neighbor[i] = np.abs(neighbor[i] - 1)

        else:
            vals = list(np.arange(self.max_val))
            vals.remove(neighbor[i])
            neighbor[i] = vals[np.random.randint(0, self.max_val - 1)]

        return neighbor

    def get_fitness(self):
        return self.fitness

    def eval_fitness(self, state):
        fitness = self.loss.evaluate(state)
        return fitness

    def random(self):
        state = np.random.randint(0, self.max_val, self.length)
        return state

    def reset(self):
        self.state = self.random()
        self.fitness = self.eval_fitness(self.state)

    def set_state(self, new_state):
        self.state = new_state
        self.fitness = self.eval_fitness(self.state)

    def get_state(self):
        return self.state

    # def get_maximize(self):
    #    return self.maximize


class LossDiscreteState:
    """Loss Function"""

    def __init__(
        self,
        label_idx,
        clf,
        x_test,
        distractor,
        cols_swap,
        reg,
        max_features=3,
        maximize=True,
    ):
        self.target = label_idx
        self.clf = clf
        self.x_test = x_test
        self.reg = reg
        self.distractor = distractor
        self.cols_swap = cols_swap  # Column names that we can swap
        self.prob_type = "discrete"
        self.max_features = 3 if max_features is None else max_features
        self.maximize = maximize
        self.window_size = x_test.shape[-1]
        self.channels = x_test.shape[-2]

    def __call__(self, feature_matrix):
        return self.evaluate(feature_matrix)

    def evaluate(self, feature_matrix):
        new_case = self.x_test.copy()
        assert len(self.cols_swap) == len(feature_matrix)

        for col_replace, a in zip(self.cols_swap, feature_matrix):
            if a == 1:
                new_case[0][col_replace] = self.distractor[0][col_replace]

        replaced_feature_count = np.sum(feature_matrix)

        input_ = new_case
        result_org = self.clf(input_)
        result = result_org[0][self.target]
        feature_loss = self.reg * np.maximum(
            0, replaced_feature_count - self.max_features
        )
        loss_pred = np.square(np.maximum(0, 0.95 - result))
        loss_pred = loss_pred + feature_loss

        return loss_pred
