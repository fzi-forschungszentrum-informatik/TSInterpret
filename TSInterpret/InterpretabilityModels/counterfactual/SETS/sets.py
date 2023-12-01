# Author: Omar Bahri

import itertools
import os
import random
import sys

import numpy as np
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from tslearn.neighbors import KNeighborsTimeSeries

from TSInterpret.InterpretabilityModels.counterfactual.SETS.utils import (
    get_nearest_neighbor,
    get_shapelets_locations_test,
)


def sets_explain(
    instance_idx, 
    model,
    data,
    ts_length,
    st_shapelets,
    all_shapelet_locations,
    all_shapelet_locations_test,
    all_shapelets_class,
    all_heat_maps,
    all_shapelets_scores,
):
    random.seed(42)

    X_train, y_train, X_test, y_test = data

    # Sort dimensions by their highest shapelet scores
    shapelets_best_scores = []
    for dim in range(len(st_shapelets)):
        shapelets_best_scores.append(max(all_shapelets_scores[dim]))

    shapelets_best_scores = np.argsort(shapelets_best_scores)[::-1]

    from sklearn.neural_network import MLPClassifier

    # train black-box model and get predictions
    model = keras.models.load_model("./best_model.hdf5")
    
    print(X_test.shape)
    y_pred = model.predict(np.swapaxes(X_test, 1, 2))
    y_pred = np.argmax(y_pred, axis=1)

    # value ranges of shapelet-classes (used later for scaling)
    all_shapelets_ranges = {}
    for c in np.unique(y_train):
        all_shapelets_ranges[c] = []

    for dim in range(X_test.shape[1]):
        shapelets_ranges = {}
        for c in np.unique(y_train):
            print(c)
            shapelets_ranges[c] = {}

        # Get [min,max] of each shapelet occurences
        for label, (all_shapelets_class, shapelets_ranges) in enumerate(
            zip(list(all_shapelets_class.values()), list(shapelets_ranges.values()))
        ):
            for j, sls in enumerate(
                [all_shapelet_locations[dim][i] for i in all_shapelets_class[dim]]
            ):
                s_mins = 0
                s_maxs = 0
                n_s = 0

                for sl in sls:
                    ts = X_train[sl[0]][dim][sl[1] : sl[2]]

                    s_mins += ts.min()
                    s_maxs += ts.max()
                    n_s += 1

                # print(all_shapelets_class[dim][j])

                shapelets_ranges[all_shapelets_class[dim][j]] = (
                    s_mins / n_s,
                    s_maxs / n_s,
                )

            for c in np.unique(y_train):
                all_shapelets_ranges[c] = shapelets_ranges

    # dictionary to store class KNNs
    knns = {}

    # fit a KNN for each class
    for c in np.unique(y_train):
        knns[c] = KNeighborsTimeSeries(n_neighbors=1)
        X_train_knn = X_train[np.argwhere(y_train == c)].reshape(
            np.argwhere(y_train == c).shape[0], X_train.shape[1], X_train.shape[2]
        )
        X_train_knn = np.swapaxes(X_train_knn, 1, 2)
        knns[c].fit(X_train_knn)

        
    orig_c = y_pred[instance_idx]

    for target_c in set(np.unique(y_train)) - set([orig_c]):
        # print("instance_idx: " + str(instance_idx))
        # print("from: " + str(orig_c) + " to: " + str(target_c))

        # get the original class shapelets and ranges
        print(all_shapelets_class)
        original_all_shapelets_class = all_shapelets_class[dim][orig_c]
        original_shapelets_ranges = all_shapelets_ranges[c]

        # get the target class shapelets and ranges
        all_target_shapelets_class = all_shapelets_class[dim][target_c]
        all_target_heat_maps = all_heat_maps[target_c]
        target_knn = knns[target_c]
        target_shapelets_ranges = all_shapelets_ranges[target_c]

        nn_idx = get_nearest_neighbor(
            target_knn,
            X_test,
            y_test,
            y_train,
            # X_test[y_test == target_c],
            # y_test[y_test == target_c],
            # y_train[y_train == target_c],
            instance_idx,
        )
        cf_dims = np.zeros((len(shapelets_best_scores), ts_length))

        # starting the with the most important dimension, start CF generation
        for dim in shapelets_best_scores:
            cf = X_test[instance_idx].copy()
            print("CF", np.expand_dims(np.swapaxes(cf, 0, 1), axis=0).shape)
            cf_pred = model.predict(np.expand_dims(np.swapaxes(cf, 0, 1), axis=0))
            cf_pred = np.argmax(cf_pred)
            if y_pred[instance_idx] == cf_pred:
                # Get the locations where the original class shapelets occur
                all_locs = get_shapelets_locations_test(
                    instance_idx,
                    all_shapelet_locations_test,
                    dim,
                    original_all_shapelets_class,
                )

                # Replace the original class shapelets with nn values
                for c_i in all_locs:
                    for loc in all_locs.get(c_i):
                        cf_pred = model.predict(
                            np.expand_dims(np.swapaxes(cf, 0, 1), axis=0)
                        )
                        cf_pred = np.argmax(cf_pred)
                        if y_pred[instance_idx] == cf_pred:
                            # print('Removing original shapelet')
                            nn = X_test[nn_idx].reshape(-1)

                            target_shapelet = nn[loc[0] : loc[1]]

                            s_min = target_shapelet.min()
                            s_max = target_shapelet.max()
                            t_min = cf[dim][loc[0] : loc[1]].min()
                            t_max = cf[dim][loc[0] : loc[1]].max()

                            if s_max - s_min == 0:
                                target_shapelet = (
                                    (t_max + t_min) / 2 * np.ones(len(target_shapelet))
                                )
                            else:
                                target_shapelet = (t_max - t_min) * (
                                    target_shapelet - s_min
                                ) / (s_max - s_min) + t_min

                            start = loc[0]
                            end = loc[1]

                            cf[dim][start:end] = target_shapelet

                # Introduce new shapelets from the target class
                for _, target_shapelet_idx in enumerate(all_target_heat_maps[dim]):
                    cf_pred = model.predict(
                        np.expand_dims(np.swapaxes(cf, 0, 1), axis=0)
                    )
                    cf_pred = np.argmax(cf_pred)
                    if y_pred[instance_idx] == cf_pred:
                        # print('Introducing new shapelet')
                        h_m = all_target_heat_maps[dim].get(target_shapelet_idx)

                        center = (
                            np.argwhere(h_m > 0)[-1][0] - np.argwhere(h_m > 0)[0][0]
                        ) // 2 + np.argwhere(h_m > 0)[0][0]

                        target_shapelet = st_shapelets[dim][target_shapelet_idx][0]
                        target_shapelet_length = target_shapelet.shape[0]

                        start = center - target_shapelet_length // 2
                        end = center + (
                            target_shapelet_length - target_shapelet_length // 2
                        )

                        if start < 0:
                            end = end - start
                            start = 0

                        if end > ts_length:
                            start = start - (end - ts_length + 1)
                            end = ts_length - 1

                        s_min = target_shapelet.min()
                        s_max = target_shapelet.max()
                        t_min = cf[dim][start:end].min()
                        t_max = cf[dim][start:end].max()

                        if s_max - s_min == 0:
                            target_shapelet = (
                                (t_max + t_min) / 2 * np.ones(len(target_shapelet))
                            )
                        else:
                            target_shapelet = (t_max - t_min) * (
                                target_shapelet - s_min
                            ) / (s_max - s_min) + t_min

                        cf[dim][start:end] = target_shapelet

            # Save the perturbed dimension
            cf_dims[dim] = cf[dim]

            cf_pred = model.predict(np.expand_dims(np.swapaxes(cf, 0, 1), axis=0))
            cf_pred = np.argmax(cf_pred)
            # if a CF is found, save it and move to next time series instance
            # print(y_pred[instance_idx], cf_pred)
            if y_pred[instance_idx] != cf_pred:
                # print('cf found')
                return cf, y_pred[instance_idx]

            else:
                # print("Trying dims combinations")
                # Try all combinations of dimensions
                for L in range(0, len(shapelets_best_scores) + 1):
                    for subset in itertools.combinations(shapelets_best_scores, L):
                        if len(subset) >= 2:
                            cf = X_test[instance_idx].copy()
                            for dim_ in subset:
                                cf[dim_] = cf_dims[dim_]

                            cf_pred = model.predict(
                                np.expand_dims(np.swapaxes(cf, 0, 1), axis=0)
                            )
                            cf_pred = np.argmax(cf_pred)
                            if y_pred[instance_idx] != cf_pred:
                                # print('cf found')
                                # print('final dims: ', subset)
                                break

            # if a CF is found, save it and move to next time series instance
            if y_pred[instance_idx] != cf_pred:
                # print('cf found')
                return cf, y_pred[instance_idx]

            else:
                print("No Counterfactual could be found this data instance")
                return None


if __name__ == "__main__":
    main()
