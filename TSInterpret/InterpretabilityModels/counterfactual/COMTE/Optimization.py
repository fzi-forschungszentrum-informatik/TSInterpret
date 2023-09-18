import logging
import multiprocessing
import numbers
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from skopt import gbrt_minimize, gp_minimize
from TSInterpret.InterpretabilityModels.counterfactual.COMTE.Problem import (
    LossDiscreteState,
    Problem,
)
from TSInterpret.InterpretabilityModels.counterfactual.COMTE.Optimization_helpers import (
    random_hill_climb,
)


class BaseExplanation:
    def __init__(
        self,
        clf,
        timeseries,
        labels,
        silent=True,
        num_distractors=2,
        dont_stop=False,
        threads=multiprocessing.cpu_count(),
    ):
        self.clf = clf
        self.timeseries = timeseries
        self.labels = pd.DataFrame(labels, columns=["label"])
        self.silent = silent
        self.num_distractors = num_distractors
        self.dont_stop = dont_stop
        self.window_size = timeseries.shape[-1]
        self.channels = timeseries.shape[-2]
        self.ts_min = np.repeat(timeseries.min(), self.window_size)
        self.ts_max = np.repeat(timeseries.max(), self.window_size)
        self.ts_std = np.repeat(timeseries.std(), self.window_size)
        self.tree = None
        self.per_class_trees = None
        self.threads = threads

    def explain(self, x_test, **kwargs):
        raise NotImplementedError("Please don't use the base class directly")

    def construct_per_class_trees(self):
        if self.per_class_trees is not None:
            return
        self.per_class_trees = {}
        self.per_class_node_indices = {c: [] for c in np.unique(self.labels)}
        input_ = self.timeseries

        preds = np.argmax(self.clf(input_), axis=1)
        true_positive_node_ids = {c: [] for c in np.unique(self.labels)}
        for pred, (idx, row) in zip(preds, self.labels.iterrows()):
            if row["label"] == pred:
                true_positive_node_ids[pred].append(idx)
        for c in np.unique(self.labels):
            dataset = []
            for node_id in true_positive_node_ids[c]:
                dataset.append(self.timeseries[[node_id], :, :].T.flatten())
                self.per_class_node_indices[c].append(node_id)
            if len(dataset) != 0:
                self.per_class_trees[c] = KDTree(np.stack(dataset))
            else:
                self.per_class_trees[c] = []
                logging.warning(
                    f"Due to lack of true postitives for class {c} no kd-tree could be build."
                )

    def construct_tree(self):
        if self.tree is not None:
            return
        train_set = []
        self.node_indices = []
        for node_id in self.timeseries.index.get_level_values("node_id").unique():
            train_set.append(self.timeseries[[node_id], :, :].T.flatten())
            self.node_indices.append(node_id)
        self.tree = KDTree(np.stack(train_set))

    def _get_distractors(self, x_test, to_maximize, n_distractors=2):
        self.construct_per_class_trees()
        # to_maximize can be int, string or np.int64
        if isinstance(to_maximize, numbers.Integral):
            to_maximize = np.unique(self.labels)[to_maximize]
        distractors = []
        for idx in (
            self.per_class_trees[to_maximize]
            .query(x_test.T.flatten().reshape(1, -1), k=n_distractors)[1]
            .flatten()
        ):
            distractors.append(
                self.timeseries[[self.per_class_node_indices[to_maximize][idx]], :, :]
            )

        return distractors

    def local_lipschitz_estimate(
        self,
        x,
        optim="gp",
        eps=None,
        bound_type="box",
        clip=True,
        n_calls=100,
        njobs=-1,
        verbose=False,
        exp_kwargs=None,
        n_neighbors=None,
    ):
        np_x = x.T.flatten()
        if n_neighbors is not None and self.tree is None:
            self.construct_tree()

        # Compute bounds for optimization
        if eps is None:
            # If want to find global lipzhitz ratio maximizer
            # search over "all space" - use max min bounds of dataset
            # fold of interest
            lwr = self.ts_min.flatten()
            upr = self.ts_max.flatten()
        elif bound_type == "box":
            lwr = (np_x - eps).flatten()
            upr = (np_x + eps).flatten()
        elif bound_type == "box_std":
            lwr = (np_x - eps * self.ts_std).flatten()
            upr = (np_x + eps * self.ts_std).flatten()
        if clip:
            lwr = lwr.clip(min=self.ts_min.min())
            upr = upr.clip(max=self.ts_max.max())
        if exp_kwargs is None:
            exp_kwargs = {}

        consts = []
        bounds = []
        variable_indices = []
        for idx, (l, u) in enumerate(zip(*[lwr, upr])):
            if u == l:
                consts.append(l)
            else:
                consts.append(None)
                bounds.append((l, u))
                variable_indices.append(idx)
        consts = np.array(consts)
        variable_indices = np.array(variable_indices)

        orig_explanation = set(self.explain(x, **exp_kwargs))
        if verbose:
            logging.info("Original explanation: %s", orig_explanation)

        def lipschitz_ratio(y):
            nonlocal self
            nonlocal consts
            nonlocal variable_indices
            nonlocal orig_explanation
            nonlocal np_x
            nonlocal exp_kwargs

            if len(y) == len(consts):
                # For random search
                consts = y
            else:
                # Only search in variables that vary
                np.put_along_axis(consts, variable_indices, y, axis=0)
            df_y = pd.DataFrame(
                np.array(consts).reshape((len(self.metrics), self.window_size)).T,
                columns=self.metrics,
            )
            df_y = pd.concat([df_y], keys=["y"], names=["node_id"])
            new_explanation = set(self.explain(df_y, **exp_kwargs))
            # Hamming distance
            exp_distance = len(orig_explanation.difference(new_explanation)) + len(
                new_explanation.difference(orig_explanation)
            )
            # Multiply by 1B to get a sensible number
            ratio = exp_distance * -1e9 / np.linalg.norm(np_x - consts)
            if verbose:
                logging.info("Ratio: %f", ratio)
            return ratio

        # Run optimization
        min_ratio = 0
        worst_case = np_x
        if n_neighbors is not None:
            for idx in self.tree.query(np_x.reshape(1, -1), k=n_neighbors)[1].flatten():
                y = self.timeseries[[self.node_indices[idx]], :, :].T.flatten()
                ratio = lipschitz_ratio(y)
                if ratio < min_ratio:
                    min_ratio = ratio
                    worst_case = y
            if verbose:
                logging.info("The worst case explanation was for %s", idx)
        elif optim == "gp":
            logging.info("Running BlackBox Minimization with Bayesian Opt")
            # Need minus because gp only has minimize method
            res = gp_minimize(
                lipschitz_ratio, bounds, n_calls=n_calls, verbose=verbose, n_jobs=njobs
            )
            min_ratio, worst_case = res["fun"], np.array(res["x"])
        elif optim == "gbrt":
            logging.info("Running BlackBox Minimization with GBT")
            res = gbrt_minimize(
                lipschitz_ratio, bounds, n_calls=n_calls, verbose=verbose, n_jobs=njobs
            )
            min_ratio, worst_case = res["fun"], np.array(res["x"])
        elif optim == "random":
            for i in range(n_calls):
                y = (upr - lwr) * np.random.random(len(np_x)) + lwr
                ratio = lipschitz_ratio(y)
                if ratio < min_ratio:
                    min_ratio = ratio
                    worst_case = y

        if len(worst_case) != len(consts):
            np.put_along_axis(consts, variable_indices, worst_case, axis=0)

        return min_ratio, consts


CLASSIFIER = None
X_TEST = None
DISTRACTOR = None


def _eval_one(tup):
    column, label_idx = tup
    global CLASSIFIER
    global X_TEST
    global DISTRACTOR
    x_test = X_TEST.copy()
    x_test[0][column] = DISTRACTOR[0][column]
    input_ = x_test.reshape(1, -1, x_test.shape[-1])

    return CLASSIFIER(input_)[0][
        label_idx
    ]  # CLASSIFIER.predict_proba(x_test)[0][label_idx]


class BruteForceSearch(BaseExplanation):
    def _find_best(self, x_test, distractor, label_idx):
        global CLASSIFIER
        global X_TEST
        global DISTRACTOR
        CLASSIFIER = self.clf
        X_TEST = x_test
        DISTRACTOR = distractor
        input_ = x_test
        best_case = self.clf(input_)[0][label_idx]
        best_column = None
        tuples = []
        for c in range(0, self.channels):
            if np.any(distractor[0][c] != x_test[0][c]):
                tuples.append((c, label_idx))
        if self.threads == 1:
            results = []
            for t in tuples:
                results.append(_eval_one(t))
        else:
            pool = multiprocessing.Pool(self.threads)
            results = pool.map(_eval_one, tuples)
            pool.close()
            pool.join()
        for (c, _), pred in zip(tuples, results):
            if pred > best_case:
                best_column = c
                best_case = pred
        if not self.silent:
            logging.info("Best column: %s, best case: %s", best_column, best_case)
        return best_column, best_case

    def explain(self, x_test, to_maximize=None, num_features=10):
        input_ = x_test
        orig_preds = self.clf(input_)
        if to_maximize is None:
            to_maximize = np.argsort(orig_preds)[0][-2:-1][0]

        orig_label = np.argmax(self.clf(input_))
        if orig_label == to_maximize:
            print("Original and Target Label are identical !")
            return None, None
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors
        )
        # print('distractores',distractors)
        best_explanation = set()
        best_explanation_score = 0
        for count, dist in enumerate(distractors):
            explanation = []
            modified = x_test.copy()
            prev_best = 0
            # best_dist = dist
            while True:
                input_ = modified
                probas = self.clf(input_)
                if np.argmax(probas) == to_maximize:
                    current_best = np.max(probas)
                    if current_best > best_explanation_score:
                        best_explanation = explanation
                        best_explanation_score = current_best
                    if current_best <= prev_best:
                        break
                    prev_best = current_best
                    if not self.dont_stop:
                        break
                if (
                    not self.dont_stop
                    and len(best_explanation) != 0
                    and len(explanation) >= len(best_explanation)
                ):
                    break
                best_column, _ = self._find_best(modified, dist, to_maximize)
                if best_column is None:
                    break

                modified[0][best_column] = dist[0][best_column]
                explanation.append(best_column)

        other = modified
        target = np.argmax(self.clf(other), axis=1)
        return other, target


class OptimizedSearch(BaseExplanation):
    def __init__(
        self,
        clf,
        timeseries,
        labels,
        silent,
        threads,
        num_distractors,
        max_attempts,
        maxiter,
        **kwargs,
    ):
        super().__init__(clf, timeseries, labels, **kwargs)
        self.discrete_state = False
        self.backup = BruteForceSearch(clf, timeseries, labels, **kwargs)
        self.max_attemps = max_attempts
        self.maxiter = maxiter

    def opt_Discrete(self, to_maximize, x_test, dist, columns, init, num_features=None):
        fitness_fn = LossDiscreteState(
            to_maximize,
            self.clf,
            x_test,
            dist,
            columns,
            reg=0.8,
            max_features=num_features,
            maximize=False,
        )
        problem = Problem(length=len(columns), loss=fitness_fn, max_val=2)
        best_state, best_fitness = random_hill_climb(
            problem,
            max_attempts=self.max_attemps,
            max_iters=self.maxiter,
            init_state=init,
            restarts=5,
        )

        self.discrete_state = True
        return best_state

    def _prune_explanation(
        self, explanation, x_test, dist, to_maximize, max_features=None
    ):
        if max_features is None:
            max_features = len(explanation)
        short_explanation = set()
        while len(short_explanation) < max_features:
            modified = x_test.copy()
            for c in short_explanation:
                modified[0][c] = dist[0][c]
            input_ = modified
            prev_proba = self.clf(input_)[0][to_maximize]
            best_col = None
            best_diff = 0
            for c in explanation:
                tmp = modified.copy()

                tmp[0][c] = dist[0][c]
                input_ = tmp
                cur_proba = self.clf(input_)[0][to_maximize]
                if cur_proba - prev_proba > best_diff:
                    best_col = c
                    best_diff = cur_proba - prev_proba
            if best_col is None:
                break
            else:
                short_explanation.add(best_col)
        return short_explanation

    def explain(
        self, x_test, num_features=None, to_maximize=None
    ) -> Tuple[np.array, int]:
        input_ = x_test
        orig_preds = self.clf(input_)

        orig_label = np.argmax(orig_preds)

        if to_maximize is None:
            to_maximize = np.argsort(orig_preds)[0][-2:-1][0]

        if orig_label == to_maximize:
            print("Original and Target Label are identical !")
            return None, None

        explanation = self._get_explanation(x_test, to_maximize, num_features)
        tr, _ = explanation
        if tr is None:
            print("Run Brute Force as Backup.")
            explanation = self.backup.explain(
                x_test, num_features=num_features, to_maximize=to_maximize
            )
        best, other = explanation
        target = np.argmax(self.clf(best), axis=1)

        return best, target

    def _get_explanation(self, x_test, to_maximize, num_features):
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors
        )

        # Avoid constructing KDtrees twice
        self.backup.per_class_trees = self.per_class_trees
        self.backup.per_class_node_indices = self.per_class_node_indices

        best_explanation = set()
        best_explanation_score = 0

        for count, dist in enumerate(distractors):
            columns = [
                c for c in range(0, self.channels) if np.any(dist[0][c] != x_test[0][c])
            ]

            # Init options
            init = [0] * len(columns)

            result = self.opt_Discrete(
                to_maximize, x_test, dist, columns, init=init, num_features=num_features
            )

            if not self.discrete_state:
                explanation = {
                    x for idx, x in enumerate(columns) if idx in np.nonzero(result.x)[0]
                }
            else:
                explanation = {
                    x for idx, x in enumerate(columns) if idx in np.nonzero(result)[0]
                }

            explanation = self._prune_explanation(
                explanation, x_test, dist, to_maximize, max_features=num_features
            )

            modified = x_test.copy()

            for c in columns:
                if c in explanation:
                    modified[0][c] = dist[0][c]
            input_ = modified  # .reshape(1, -1, self.window_size)
            probas = self.clf(input_)

            if not self.silent:
                logging.info("Current probas: %s", probas)
            if np.argmax(probas) == to_maximize:
                current_best = np.max(probas)
                if current_best > best_explanation_score:
                    best_explanation = explanation
                    best_explanation_score = current_best
                    best_modified = modified

        if len(best_explanation) == 0:
            return None, None

        return best_modified, best_explanation
