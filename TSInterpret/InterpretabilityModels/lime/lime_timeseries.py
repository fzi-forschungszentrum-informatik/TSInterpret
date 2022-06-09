import numpy as np
import sklearn
from lime import explanation
from lime import lime_base
import math
import logging
import matplotlib.pyplot as plt
from InterpretabilityModels.InterpretabilityBase import InterpretabilityMethod
class TSDomainMapper(explanation.DomainMapper):
    def __init__(self, signal_names, num_slices, is_multivariate):
        """Init function.
        Args:
            signal_names: list of strings, names of signals
        """
        self.num_slices = num_slices
        self.signal_names = signal_names
        self.is_multivariate = is_multivariate

    def map_exp_ids(self, exp, **kwargs):
        # in case of univariate, don't change feature ids
        if not self.is_multivariate:
            return exp

        names = []
        for _id, weight in exp:
            # from feature idx, extract both the pair number of slice
            # and the signal perturbed
            nsignal = int(_id / self.num_slices)
            nslice = _id % self.num_slices
            signalname = self.signal_names[nsignal]
            featurename = "%d - %s" % (nslice, signalname)
            names.append((featurename, weight))
        return names


class LimeTimeSeriesExplainer(InterpretabilityMethod):
    """Explains time series classifiers."""

    def __init__(self, ml_model,
                 num_slices,
                 kernel_width=25,
                 num_features=10,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 signal_names=["not specified"]
                 ):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
            classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
            signal_names: list of strings, names of signals

         TODO This section is still to do
    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.
        * "num": int, default: 1
            Number of counterfactuals per factual to generate
        * "desired_class": int, default: 1
            Given a binary class label, the desired class a counterfactual should have (e.g., 0 or 1)
        * "posthoc_sparsity_param": float, default: 0.1
            Fraction of post-hoc preprocessing steps.
    - Restrictions:
        *   Only the model agnostic approach (backend: sklearn) is used in our implementation.
        *   ML model needs to have a transformation pipeline for normalization, encoding and feature order.
            See pipelining at carla/models/catalog/catalog.py for an example ML model class implementation
    .. [1] R. K. Mothilal, Amit Sharma, and Chenhao Tan. 2020. Explaining machine learning classifiers
            through diverse counterfactual explanations
        """

        # exponential kernel
        def kernel(d): return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.model_to_explain = ml_model
        self.base = lime_base.LimeBase(kernel, verbose)
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.signal_names = signal_names
        self.num_features=num_features
        self.num_slices = num_slices

    def explain(self,
                         instance,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_samples=5000,
                         model_regressor=None,
                         replacement_method='mean'):
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).
        As distance function DTW metric is used.
        Args:
            time_series_instance: time series to be explained.
            classifier_fn: classifier prediction probability function,
                which takes a list of d arrays with time series values
                and outputs a (d, k) numpy array with prediction
                probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            num_slices: Defines into how many slices the time series will
                be split up
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
            the K labels with highest prediction probabilities, where K is
            this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter to
                model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
       """

        permutations, predictions, distances = self.__data_labels_distances(
            instance, classifier_fn,
            num_samples, self.num_slices, replacement_method)
        self.window=len(instance)

        is_multivariate = len(instance.shape) > 1
        print('Pred',predictions.ndim)
        if predictions.ndim==1:
            print('No Probabilities')
            onehot = sklearn.preprocessing.OneHotEncoder()
            predictions = onehot.fit_transform(predictions.reshape(-1,1))
            print('Pred', predictions.ndim)
            print('Pred Shape', predictions.shape)


        if self.class_names is None:
            self.class_names = [str(x) for x in range(predictions[0].shape[0])]

        domain_mapper = TSDomainMapper(self.signal_names, self.num_slices, is_multivariate)
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = predictions[0]

        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels:]
            ret_exp.top_labels = list(predictions)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[int(label)],
             ret_exp.local_exp[int(label)],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                permutations, predictions,
                distances, label,
                self.num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp
    def plot_on_Sample(self,series,exp, data = (None,None),save=None):
        values_per_slice = math.ceil(len(series) / self.num_slices)
        plt.plot(series, color='b', label='Explained instance')
        #TODO Is this actually correct working
        x,y=data
        if type(x) != None:
            for num in np.unique(y):
                number=np.where(y==num)
                result =np.take(x, number, axis=0).reshape(-1,self.window)
                plt.plot(result.mean(axis=0),label=f'Mean of class {num}')
        plt.legend(loc='lower left')

        for i in range(self.num_features):
            feature, weight = exp.as_list()[i]
            print(feature)
            #TODO CHanged Feature to i
            start = i * values_per_slice
            print(start)
            end = start + values_per_slice
            color = 'red' if weight < 0 else 'green'
            plt.axvspan(start, end, color=color, alpha=abs(weight * 2))
        if save == None:
            plt.show()
            plt.close()
        else:
            plt.savefig(f'./Results/{save}/Lime4Time.png')

    def __data_labels_distances(cls,
                                timeseries,
                                classifier_fn,
                                num_samples,
                                num_slices,
                                replacement_method='mean'):
        """Generates a neighborhood around a prediction.
        Generates neighborhood data by randomly removing slices from the
        time series and replacing them with other data points (specified by
        replacement_method: mean over slice range, mean of entire series or
        random noise). Then predicts with the classifier.
        Args:
            timeseries: Time Series to be explained.
                it can be a flat array (univariate)
                or (num_signals, num_points) (multivariate)
            classifier_fn: classifier prediction probability function, which
                takes a time series and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear
                model (perturbation + original time series)
            num_slices: how many slices the time series will be split into
                for discretization.
            replacement_method:  Defines how individual slice will be
                deactivated (can be 'mean', 'total_mean', 'noise')
        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of slices in the time series. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: distance between the original instance and
                    each perturbed instance
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0].reshape([1, -1]), metric='cosine').ravel() * 100

        num_channels = 1
        len_ts = len(timeseries)
        if len(timeseries.shape) > 1:  # multivariate
            #TODO Reshape added
            timeseries=timeseries.reshape(timeseries.shape[1],timeseries.shape[0])
            num_channels,len_ts = timeseries.shape

        values_per_slice = math.ceil(len_ts / num_slices)
        deact_per_sample = np.random.randint(1, num_slices + 1, num_samples - 1)
        perturbation_matrix = np.ones((num_samples, num_channels, num_slices))
        features_range = range(num_slices)
        original_data = [timeseries.copy()]

        for i, num_inactive in enumerate(deact_per_sample, start=1):
            logging.info("sample %d, inactivating %d", i, num_inactive)
            # choose random slices indexes to deactivate
            inactive_idxs = np.random.choice(features_range, num_inactive,
                                             replace=False)
            num_channels_to_perturb = np.random.randint(1, num_channels + 1)

            channels_to_perturb = np.random.choice(range(num_channels),
                                                   num_channels_to_perturb,
                                                   replace=False)

            logging.info("sample %d, perturbing signals %r", i,
                         channels_to_perturb)

            for chan in channels_to_perturb:
                perturbation_matrix[i, chan, inactive_idxs] = 0

            tmp_series = timeseries.copy()

            for idx in inactive_idxs:
                start_idx = idx * values_per_slice
                end_idx = start_idx + values_per_slice
                end_idx = min(end_idx, len_ts)

                if replacement_method == 'mean':
                    # use mean of slice as inactive
                    perturb_mean(tmp_series, start_idx, end_idx,
                                 channels_to_perturb)
                elif replacement_method == 'noise':
                    # use random noise as inactive
                    perturb_noise(tmp_series, start_idx, end_idx,
                                  channels_to_perturb)
                elif replacement_method == 'total_mean':
                    # use total series mean as inactive
                    perturb_total_mean(tmp_series, start_idx, end_idx,
                                       channels_to_perturb)
            original_data.append(tmp_series)
        #TODO changes were done here
        #print(type(classifier_fn))
        #if type(classifier_fn).startswith('sklearn'):
        try:
            print('This is try')
            original_data = np.array(original_data).reshape(-1, timeseries.shape[1], timeseries.shape[0])
            predictions = classifier_fn(np.array(original_data))

        except:
            print('This is except')
            print(np.array(original_data).shape)
            original_data = np.array(original_data).reshape(-1, timeseries.shape[1])
            print(original_data.shape)
            predictions = classifier_fn(np.array(original_data))
            print(predictions)
        # create a flat representation for features
        perturbation_matrix = perturbation_matrix.reshape((num_samples, num_channels * num_slices))
        distances = distance_fn(perturbation_matrix)

        return perturbation_matrix, predictions, distances


def perturb_total_mean(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = m.mean()
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = m[chan].mean()


def perturb_mean(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = np.mean(m[start_idx:end_idx])
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = np.mean(m[chan][start_idx:end_idx])


def perturb_noise(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = np.random.uniform(m.min(), m.max(),
                                                 end_idx - start_idx)
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = np.random.uniform(m[chan].min(),
                                                       m[chan].max(),
                                                       end_idx - start_idx)
#def explain(series, knn, num_features, num_slices,replacement_method='total_mean',num_samples=5000,class_names=None, save=None):
    #explainer = LimeTimeSeriesExplainer()
    #exp = explainer.explain_instance(series, knn, num_features=num_features, num_samples=num_samples,num_slices=num_slices, replacement_method=replacement_method)
    #exp.as_pyplot_figure()
    #return exp

