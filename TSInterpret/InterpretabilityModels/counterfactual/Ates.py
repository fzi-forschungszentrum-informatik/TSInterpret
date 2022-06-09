#TODO 
"""Contains the code for ICAPAI'21 paper "Counterfactual Explanations for Multivariate Time Series"
Authors:
    Emre Ates (1), Burak Aksar (1), Vitus J. Leung (2), Ayse K. Coskun (1)
Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories
This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.
"""

from cProfile import label
import logging
import random
import torch
import numbers
import multiprocessing
import uuid
from typing import List, Optional,Tuple

from skopt import gp_minimize, gbrt_minimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
#Workaround for mlrose package
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from TSInterpret.InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
from TSInterpret.InterpretabilityModels.counterfactual.CF import CF
#from InterpretabilityModels.utils import torch_wrapper, tensorflow_wrapper,sklearn_wrapper
from TSInterpret.Models.PyTorchModel import PyTorchModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel
from TSInterpret.Models.SklearnModel import SklearnModel


class BaseExplanation:
    def __init__(self, clf, timeseries, labels, silent=True,
                 num_distractors=2, dont_stop=False,
                 threads=multiprocessing.cpu_count()):
        self.clf = clf
        self.timeseries = timeseries
        self.labels = pd.DataFrame(labels, columns=['label'])
        #print(self.labels)
        self.silent = silent
        self.num_distractors = num_distractors
        #self.metrics = self.clf.steps[0][1].column_names
        self.dont_stop = dont_stop
        self.window_size =timeseries.shape[-1] #len(timeseries.loc[
            #timeseries.index.get_level_values('node_id')[0]])
        self.channels=timeseries.shape[-2]
        self.ts_min = np.repeat(timeseries.min(), self.window_size)
        self.ts_max = np.repeat(timeseries.max(), self.window_size)
        self.ts_std = np.repeat(timeseries.std(), self.window_size)
        self.tree = None
        self.per_class_trees = None
        self.threads = threads

    def explain(self, x_test, **kwargs):
        raise NotImplementedError("Please don't use the base class directly")

    #def _get_feature_names(self, clf, timeseries):
    #    if hasattr(self.clf.steps[1][1], 'transform'):
    #        return self.clf.steps[2][1].column_names
    #    else:
    #        window_size = len(timeseries.loc[
    #            [timeseries.index.get_level_values('node_id')[0]], :, :])
    #        names = []
    #        for c in timeseries.columns:
    #            for i in range(window_size):
    #                names.append(c + '_' + str(i) + 's')
    #       return names

    #def _transform_data(self, data, sample=None):
    #    if hasattr(self.clf.steps[1][1], 'transform'):
    #        transformed = self.clf.steps[1][1].transform(data)
    #        if sample:
    #            transformed = transformed.sample(sample)
    #        return self.clf.steps[3][1].transform(transformed)
    #    else:
            # autoencoder
    #        train_set = []
    #        for node_id in data.index.get_level_values('node_id').unique():
    #            train_set.append(data.loc[[node_id], :, :].values.T.flatten())
    #        result = np.stack(train_set)
    #        if sample:
    #            idx = np.random.randint(len(result), size=sample)
    #            result = result[idx, :]
    #        return result

    def _plot_changed(self, metric, original, distractor, savefig=False):
        fig = plt.figure(figsize=(6,3))
        ax = fig.gca()
        #print(distractor.shape)
        #print(original.shape)
        plt.plot(range(distractor.shape[-1]),
                 original[metric], label='x$_{test}$',
                 figure=fig,
                 )
        plt.plot(range(distractor.shape[-1]),
                 distractor[0][metric], label='Distractor',
                 figure=fig)
        ax.set_ylabel(metric)
        ax.set_xlabel('Time (s)')
        ax.legend()
        if savefig:
            filename = "{}.pdf".format(uuid.uuid4())
            fig.savefig(filename, bbox_inches='tight')
            logging.info("Saved the figure to %s", filename)
        fig.show()

    def construct_per_class_trees(self):
        """Used to choose distractors"""
        if self.per_class_trees is not None:
            return
        self.per_class_trees = {}
        self.per_class_node_indices = {c: [] for c in np.unique(self.labels)}
        #todo predict
        #print('timeseries',self.timeseries.shape)
        input_ = self.timeseries.reshape(-1,self.channels,self.window_size)
        #print('input',input_.shape)
        preds = np.argmax(self.clf(input_),axis=1)
        #print(preds)
        #print(self.labels)
        #preds = self.clf.predict(self.timeseries)
        true_positive_node_ids = {c: [] for c in np.unique(self.labels)}
        #print(type(true_positive_node_ids))
        #print('Index')
        for pred, (idx, row) in zip(preds, self.labels.iterrows()):
            
            if row['label'] == pred:
                #print(idx)
                true_positive_node_ids[pred].append(idx)
        #print(true_positive_node_ids)
        for c in  np.unique(self.labels):
            dataset = []
            #print('c',c)
            for node_id in true_positive_node_ids[c]:
                #print('NodeID',node_id)
                dataset.append(self.timeseries[
                    [node_id], :, :].T.flatten())
                #print('timeseries',self.timeseries[node_id])
                self.per_class_node_indices[c].append(node_id)
            #print(np.array(dataset).shape)
            if len(dataset)!=0:
                self.per_class_trees[c] = KDTree(np.stack(dataset))
            else: 
                self.per_class_trees[c]=[]
                logging.warning(f'Due to lack of true postitives for class {c} no kd-tree could be build.')

        if not self.silent:
            logging.info("Finished constructing per class kdtree")

    def construct_tree(self):
        if self.tree is not None:
            return
        train_set = []
        self.node_indices = []
        for node_id in self.timeseries.index.get_level_values(
                'node_id').unique():
            train_set.append(self.timeseries[
                [node_id], :, :].T.flatten())
            self.node_indices.append(node_id)
        self.tree = KDTree(np.stack(train_set))
        if not self.silent:
            logging.info("Finished constructing the kdtree")

    def _get_distractors(self, x_test, to_maximize, n_distractors=2):
        self.construct_per_class_trees()
        # to_maximize can be int, string or np.int64
        if isinstance(to_maximize, numbers.Integral):
            to_maximize =np.unique(self.labels)[to_maximize]
        distractors = []
        #print('to_maximize',to_maximize)
        #print('Class Tree',self.per_class_trees)
        #print('Class Tree with id',self.per_class_trees[to_maximize])
        for idx in self.per_class_trees[to_maximize].query(
                x_test.T.flatten().reshape(1, -1),
                k=n_distractors)[1].flatten():
            distractors.append(self.timeseries[
                [self.per_class_node_indices[to_maximize][idx]], :, :])
        if not self.silent:
            logging.info("Returning distractors %s", [
                x#.index.get_level_values('node_id').unique().values[0]
                for x in distractors])
        return distractors

    def local_lipschitz_estimate(
            self, x, optim='gp', eps=None, bound_type='box', clip=True,
            n_calls=100, njobs=-1, verbose=False, exp_kwargs=None,
            n_neighbors=None):
        """Compute one-sided lipschitz estimate for explainer.
        Adequate for local Lipschitz, for global must have
        the two sided version. This computes:
            max_z || f(x) - f(z)|| / || x - z||
        Instead of:
            max_z1,z2 || f(z1) - f(z2)|| / || z1 - z2||
        If n_neighbors is provided, does a local search on n closest neighbors
        If eps provided, does local lipzshitz in:
            - box of width 2*eps along each dimension if bound_type = 'box'
            - box of width 2*eps*va, along each dimension if bound_type =
                'box_norm' (i.e. normalize so that deviation is
                eps % in each dim )
            - box of width 2*eps*std along each dimension if bound_type =
                'box_std'
        max_z || f(x) - f(z)|| / || x - z||   , with f = theta
        clip: clip bounds to within (min, max) of dataset
        """
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
        elif bound_type == 'box':
            lwr = (np_x - eps).flatten()
            upr = (np_x + eps).flatten()
        elif bound_type == 'box_std':
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
            df_y = pd.DataFrame(np.array(consts).reshape((len(self.metrics),
                                                          self.window_size)).T,
                                columns=self.metrics)
            df_y = pd.concat([df_y], keys=['y'], names=['node_id'])
            new_explanation = set(self.explain(df_y, **exp_kwargs))
            # Hamming distance
            exp_distance = len(orig_explanation.difference(new_explanation)) \
                + len(new_explanation.difference(orig_explanation))
            # Multiply by 1B to get a sensible number
            ratio = exp_distance * -1e9 / np.linalg.norm(np_x - consts)
            if verbose:
                logging.info("Ratio: %f", ratio)
            return ratio

        # Run optimization
        min_ratio = 0
        worst_case = np_x
        if n_neighbors is not None:
            for idx in self.tree.query(np_x.reshape(1, -1),
                                       k=n_neighbors)[1].flatten():
                y = self.timeseries[
                    [self.node_indices[idx]], :, :].T.flatten()
                ratio = lipschitz_ratio(y)
                if ratio < min_ratio:
                    min_ratio = ratio
                    worst_case = y
            if verbose:
                logging.info("The worst case explanation was for %s", idx)
        elif optim == 'gp':
            logging.info('Running BlackBox Minimization with Bayesian Opt')
            # Need minus because gp only has minimize method
            res = gp_minimize(lipschitz_ratio, bounds, n_calls=n_calls,
                              verbose=verbose, n_jobs=njobs)
            min_ratio, worst_case = res['fun'], np.array(res['x'])
        elif optim == 'gbrt':
            logging.info('Running BlackBox Minimization with GBT')
            res = gbrt_minimize(lipschitz_ratio, bounds, n_calls=n_calls,
                                verbose=verbose, n_jobs=njobs)
            min_ratio, worst_case = res['fun'], np.array(res['x'])
        elif optim == 'random':
            for i in range(n_calls):
                y = (upr - lwr) * np.random.random(len(np_x)) + lwr
                ratio = lipschitz_ratio(y)
                if ratio < min_ratio:
                    min_ratio = ratio
                    worst_case = y

        if len(worst_case) != len(consts):
            np.put_along_axis(consts, variable_indices, worst_case, axis=0)
        if verbose:
            logging.info("Best ratio: %f, norm: %f", min_ratio,
                         np.linalg.norm(np_x - consts))
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
    input_ = x_test.reshape(1,-1,x_test.shape[-1])
    
    return CLASSIFIER(input_)[0][label_idx] #CLASSIFIER.predict_proba(x_test)[0][label_idx]


class BruteForceSearch(BaseExplanation):
    def _find_best(self, x_test, distractor, label_idx):
        global CLASSIFIER
        global X_TEST
        global DISTRACTOR
        CLASSIFIER = self.clf
        X_TEST = x_test
        DISTRACTOR = distractor
        input_ = x_test.reshape(1,-1,self.window_size)
        best_case= self.clf(input_)[0][label_idx]
        #best_case = self.clf.predict_proba(x_test)[0][label_idx]
        best_column = None
        tuples = []
        for c in range(0, self.channels):
            #print(distractor.shape)
            #print(x_test.shape)
            if np.any(distractor[0][c]!= x_test[0][c]):
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
            logging.info("Best column: %s, best case: %s",
                         best_column, best_case)
        return best_column, best_case

    def explain(self, x_test, to_maximize=None, num_features=10,return_dist=False, savefig=False):
        #orig_preds = self.clf.predict_proba(x_test)
        input_ = x_test.reshape(1,-1,self.window_size)    
        orig_preds= self.clf(input_)[0][to_maximize]
        orig_label = np.argmax(orig_preds)
        if to_maximize is None:
            to_maximize = np.argmin(orig_preds)
        if orig_label == to_maximize:
            return []
        if not self.silent:
            logging.info("Working on turning label from %s to %s",
                         orig_label, to_maximize)
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors)
        best_explanation = set()
        best_explanation_score = 0
        for count, dist in enumerate(distractors):
            if not self.silent:
                logging.info("Trying distractor %d / %d",
                             count + 1, self.num_distractors)
            explanation = []
            modified = x_test.copy()
            prev_best = 0
            best_dist = dist #TODO: Only supports one distractor
            while True:
                input_ = modified.reshape(1,-1,self.window_size)
                probas= self.clf(input_)[0][to_maximize]
                #probas = self.clf.predict_proba(modified)
                if not self.silent:
                    logging.info("Current probas: %s", probas)
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
                if (not self.dont_stop and
                        len(best_explanation) != 0 and
                        len(explanation) >= len(best_explanation)):
                    break
                best_column, _ = self._find_best(modified, dist, to_maximize)
                if best_column is None:
                    break
                #if not self.silent:
                #    self._plot_changed(best_column, modified, dist, savefig=savefig)
                modified[0][best_column] = dist[0][best_column]
                explanation.append(best_column)

        if not return_dist:
            return explanation,modified
        else:
            return explanation, best_dist,modified


class LossDiscreteState:

    def __init__(self, label_idx, clf, x_test, distractor, cols_swap, reg,
                 max_features=3, maximize=True):
        self.target = label_idx
        self.clf = clf
        self.x_test = x_test
        self.reg = reg
        self.distractor = distractor
        self.cols_swap = cols_swap  # Column names that we can swap
        self.prob_type = 'discrete'
        self.max_features = 3 if max_features is None else max_features
        self.maximize = maximize
        self.window_size= x_test.shape[-1]
        self.channels= x_test.shape[-2]

    def __call__(self, feature_matrix):
        return self.evaluate(feature_matrix)

    def evaluate(self, feature_matrix):

        new_case = self.x_test.copy()
        assert len(self.cols_swap) == len(feature_matrix)

        # If the value is one, replace from distractor
        for col_replace, a in zip(self.cols_swap, feature_matrix):
            if a == 1:
                #print(new_case.shape)
                #print(self.distractor.shape)
                new_case[0][col_replace] = self.distractor[0][col_replace]

        replaced_feature_count = np.sum(feature_matrix)

        # if replaced_feature_count > self.max_features:
        #     feature_loss = 1
        #     loss_pred = 1
        # else:
        # Will return the prob of the other class
        input_ = new_case.reshape(1,self.channels,self.window_size)
        result = self.clf(input_)[0][self.target]
        #result = self.clf.predict_proba(new_case)[0][self.target]
        feature_loss = self.reg * np.maximum(0, replaced_feature_count - self.max_features)
        loss_pred = np.square(np.maximum(0, 0.95 - result))

        loss_pred = loss_pred + feature_loss
        #print(loss_pred)
        return -loss_pred if self.maximize else loss_pred

    def get_prob_type(self):
        """ Return the problem type."""

        return self.prob_type


class OptimizedSearch(BaseExplanation):

    def __init__(self, clf, timeseries, labels, **kwargs):
        super().__init__(clf, timeseries, labels, **kwargs)
        self.discrete_state = False
        self.backup = BruteForceSearch(clf, timeseries, labels, **kwargs)

    def opt_Discrete(self, to_maximize, x_test, dist, columns, init,
                     max_attempts, maxiter, num_features=None):

        fitness_fn = LossDiscreteState(
            to_maximize,
            self.clf, x_test, dist,
            columns, reg=0.8, max_features=num_features,
            maximize=False
        )
        problem = mlrose.DiscreteOpt(
            length=len(columns), fitness_fn=fitness_fn,
            maximize=False, max_val=2)
        best_state, best_fitness = mlrose.random_hill_climb(
            problem,
            max_attempts=max_attempts,
            max_iters=maxiter,
            init_state=init,
            restarts = 5,
        )

        self.discrete_state = True
        return best_state

    def _prune_explanation(self, explanation, x_test, dist,
                           to_maximize, max_features=None):
        if max_features is None:
            max_features = len(explanation)
        short_explanation = set()
        while len(short_explanation) < max_features:
            modified = x_test.copy()
            for c in short_explanation:
                modified[0][c] = dist[0][c]
            input_ = modified.reshape(1,-1,self.window_size)
            prev_proba = self.clf(input_)[0][to_maximize]
            #prev_proba = self.clf.predict_proba(modified)[0][to_maximize]
            best_col = None
            best_diff = 0
            for c in explanation:
                tmp = modified.copy()
                #print('temp',tmp[0].shape)
                #print('dist',dist.shape)
                tmp[0][c] = dist[0][c]
                #cur_proba = self.clf.predict_proba(tmp)[0][to_maximize]

                input_ = tmp.reshape(1,self.channels,self.window_size)
                cur_proba = self.clf(input_)[0][to_maximize]
                if cur_proba - prev_proba > best_diff:
                    best_col = c
                    best_diff = cur_proba - prev_proba
            if best_col is None:
                break
            else:
                short_explanation.add(best_col)
        return short_explanation

    def explain(self, x_test, num_features=None, to_maximize=None, return_dist = False, savefig=False)-> Tuple[np.array, int]:
        # num_feature is maximum number of features
        #orig_preds = self.clf.predict_proba(x_test)
        
        input_ = x_test.reshape(1,-1,self.window_size)
        orig_preds = self.clf(input_)
        #idx = output.argmax()
        orig_label = np.argmax(orig_preds)
        #print(orig_label)
        #print(to_maximize)
        #binary classification
        if to_maximize is None:
            to_maximize = np.argmin(orig_preds)

        if orig_label == to_maximize:
            print('Original and Target Label are identical !')
            return []
        #if not self.silent:
        #    logging.info("Working on turning label from %s to %s",
        #                 self.clf.classes_[orig_label],
        #                 self.clf.classes_[to_maximize])

        explanation = self._get_explanation(
            x_test, to_maximize, num_features, return_dist, savefig=savefig)
        if not explanation:
            #logging.info("Used greedy search for %s",
            #             x_test.index.get_level_values('node_id')[0])
            explanation = self.backup.explain(x_test, num_features=num_features,
                                              to_maximize=to_maximize, return_dist=return_dist, savefig=savefig)
        best, other = explanation
        target = np.argmax(self.clf(other),axis=1)
        print('Target', target)
        #TODO Change explanation to best and change plot func !
        return other, target

    def _get_explanation(self, x_test, to_maximize, num_features, return_dist=False, savefig=False):
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors)

        # Avoid constructing KDtrees twice
        self.backup.per_class_trees = self.per_class_trees
        self.backup.per_class_node_indices = self.per_class_node_indices

        best_explanation = set()
        best_explanation_score = 0

        for count, dist in enumerate(distractors):

            if not self.silent:
                logging.info("Trying distractor %d / %d",
                             count + 1, self.num_distractors)
            #print(dist)
            #print(dist.shape)
            columns = [
                c for c in range(0,self.channels)
                if np.any(dist[0][c] != x_test[0][c])
            ]

            # Init options
            init = [0] * len(columns)

            result = self.opt_Discrete(
                to_maximize, x_test, dist, columns, init=init,
                max_attempts=1000, maxiter=1000, num_features=num_features)

            if not self.discrete_state:
                explanation = {
                    x for idx, x in enumerate(columns)
                    if idx in np.nonzero(result.x)[0]
                }
            else:
                explanation = {
                    x for idx, x in enumerate(columns)
                    if idx in np.nonzero(result)[0]
                }

            explanation = self._prune_explanation(
                explanation, x_test, dist, to_maximize, max_features=num_features)

            modified = x_test.copy()

            for c in columns:
                if c in explanation:
                    modified[0][c] = dist[0][c]
            input_ = modified.reshape(1,-1,self.window_size)
            probas = self.clf(input_)
            #probas = self.clf.predict_proba(modified)

            if not self.silent:
                logging.info("Current probas: %s", probas)

            if np.argmax(probas) == to_maximize:
                current_best = np.max(probas)
                if current_best > best_explanation_score:
                    best_explanation = explanation
                    best_explanation_score = current_best
                    best_modified = modified
                    best_dist = dist
        #print(best_explanation)
        #print(best_modified)
        #if not self.silent and len(best_explanation) != 0:
        #    for metric in best_explanation:
        #        self._plot_changed(metric, x_test, best_dist, savefig=savefig)
        if len(best_explanation) == 0:
            return None
        if return_dist == False: #or len(best_explanation) == 0:
            return best_explanation,best_modified
        else:
            return best_explanation, best_dist,


class AtesCF(CF):
    def __init__(self,mlmodel, ref, backend, mode) -> None:
        #super().__init__()
        super().__init__(mlmodel,mode)
        #self.model_to_explain = mlmodel
        #self.mode = mode 
        self.backend=backend
        test_x,test_y=ref
        shape=test_x.shape
        if mode == 'time':
            #Parse test data into (1, feat, time):
            change=True
            self.ts_length= shape[-2]
            test_x=test_x.reshape(test_x.shape[0],test_x.shape[2],test_x.shape[1])
        elif mode == 'feat':
            change=False
            self.ts_length= shape[-1]

        if backend=='PYT':
            
            self.predict=PyTorchModel(mlmodel,change).predict
        elif backend == 'TF':
            self.predict=TensorFlowModel(mlmodel,change).predict

        elif backend == 'SK':
            self.predict=SklearnModel(mlmodel,change).predict
        self.referenceset=(test_x,test_y)
         

    def explain(self, x: np.ndarray, target: int, method= 'opt')-> Tuple[np.array, int]:
        '''
        Counterfactual according to Ates.
        Args:
            x (np.array): The instance to explain.
            target (np.array): target class.
            
        Returns:
            [np.array], int: Tuple of Counterfactual and Label
    
        '''
        if self.mode != 'feat':
            x=x.reshape(-1, x.shape[-1], x.shape[-2])
            
        train_x,train_y=self.referenceset
        if len(train_y.shape)>1:
             train_y= np.argmax(train_y,axis=1)
        #print(train_x.shape)
        if method=='opt':
            opt=OptimizedSearch(self.predict, train_x, train_y, silent=False, threads=1,num_distractors=2)
            return opt.explain(x,to_maximize=target,savefig=False)
        elif method =='brute':
            opt=BruteForceSearch(self.predict, train_x,train_y)
            return opt.explain(x,to_maximize=target,savefig=False)
        

    def plot(self,item,org_label,exp,cf_label,figsize=(15,15)):
        if self.mode == 'time':
            item = item.reshape(item.shape[0],item.shape[2],item.shape[1]) 

        plt.style.use("classic")
        colors = [
            '#08F7FE',  # teal/cyan
            '#FE53BB',  # pink
            '#F5D300',  # yellow
            '#00ff41',  # matrix green
        ]
        #Figure out number changed channels
        #index= np.where(np.any(item))
        res = (item != exp).any(-1)
        ind=np.where(res[0])
        i=0
        #Draw changed channels
        fig, ax = plt.subplots(len(ind[0]),1,figsize=figsize)
        #print(ax)
        for channel in ind[0]:
            #fig,ax=plt.subplot(len(ind[0]),1,i)
    
            df = pd.DataFrame({f'Predicted: {org_label}': list(item[0][channel].flatten()),
                   f'Counterfactual: {cf_label}': list(exp[0][channel].flatten())})
    
            df.plot(marker='.', color=colors, ax=ax[i])
            # Redraw the data with low alpha and slighty increased linewidth:
            n_shades = 10
            diff_linewidth = 1.05
            alpha_value = 0.3 / n_shades
            for n in range(1, n_shades+1):
                df.plot(marker='.',
                linewidth=2+(diff_linewidth*n),
                alpha=alpha_value,
                legend=False,
                ax=ax[i],
                color=colors)

            ax[i].grid(color='#2A3459')
            plt.xlabel('Time', fontweight = 'bold', fontsize='large')
            plt.ylabel('Value', fontweight = 'bold', fontsize='large')
            i=i+1
        #plt.savefig('../Images/Initial_Example_Neon.pdf')
        plt.show()

