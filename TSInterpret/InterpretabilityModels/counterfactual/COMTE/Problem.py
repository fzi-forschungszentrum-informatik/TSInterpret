
import logging
import multiprocessing
import numbers
import sys
from typing import Tuple
import numpy as np
import pandas as pd
import six
from sklearn.neighbors import KDTree
from skopt import gbrt_minimize, gp_minimize

sys.modules["sklearn.externals.six"] = six

import mlrose
class LossDiscreteState:
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

        input_ = new_case.reshape(1, self.channels, self.window_size)
        result = self.clf(input_)[0][self.target]
        feature_loss = self.reg * np.maximum(
            0, replaced_feature_count - self.max_features
        )
        loss_pred = np.square(np.maximum(0, 0.95 - result))

        loss_pred = loss_pred + feature_loss
        # print(loss_pred)
        return -loss_pred if self.maximize else loss_pred

    def get_prob_type(self):
        return self.prob_type