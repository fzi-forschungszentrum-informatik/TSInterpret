# Author: Omar Bahri

import copy
import itertools
import random

import numpy as np
import tensorflow.keras as keras
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from tslearn.neighbors import KNeighborsTimeSeries

from TSInterpret.InterpretabilityModels.counterfactual.SETS.utils import (
    get_all_shapelet_locations_scaled_threshold,
    get_all_shapelet_locations_scaled_threshold_test,
    get_nearest_neighbor,
    get_shapelets_locations_test,
)


# cast to tf format
def to_tff(x):
    return np.expand_dims(np.swapaxes(x, 0, 1), axis=0)


def fit_shapelets(
    data,
    ts_length,
    st_shapelets,
    shapelets_distances,
    random_seed=42,
    occlusion_threshhold=1e-1,
    remove_multiclass_shapelets=True,
):
    random.seed(random_seed)
    X_train, y_train = data

    # make deep copy for reusability
    fitted_shapelets = copy.deepcopy(st_shapelets)

    # get the shapelets locations threshhold for testing
    (
        all_shapelet_locations,
        all_no_occurences,
        threshold,
    ) = get_all_shapelet_locations_scaled_threshold(
        shapelets_distances, ts_length, occlusion_threshhold / 100.0
    )
    # initialize a dictionary that stores lists of class-shapelets
    all_shapelets_class = {}
    # initialize a dictionary that stores lists of class-shapelets heatmaps
    all_heat_maps = {}

    for c in np.unique(y_train):
        all_shapelets_class[c] = []
        all_heat_maps[c] = []

    # get the shapelet classes and their heatmaps at each dimension
    for dim in range(X_train.shape[1]):
        for index in sorted(all_no_occurences[dim], reverse=True):
            del fitted_shapelets[dim][index]

        # Get shapelets class occurences
        shapelets_classes = []
        for shapelet_locations in all_shapelet_locations[dim]:
            shapelet_classes = []
            for sl in shapelet_locations:
                shapelet_classes.append(y_train[sl[0]])
            shapelets_classes.append(shapelet_classes)

        if remove_multiclass_shapelets:
            not_one_class = []
            # Find shapelets that happen exclusively under one class
            for i, shapelet_class in enumerate(shapelets_classes):
                if len(np.unique(shapelet_class)) > 1:
                    not_one_class.append(i)

            for index in sorted(not_one_class, reverse=True):
                del fitted_shapelets[dim][index]
                del all_shapelet_locations[dim][index]
                del shapelets_classes[index]

        # initialize a dictionary that stores lists of class-shapelets
        # for current dimension
        shapelets_class = {}
        # initialize a dictionary that stores class-shapelets
        # heatmaps for current dimension
        heat_maps = {}
        for c in np.unique(y_train):
            shapelets_class[c] = []
            heat_maps[c] = {}

        # keep shapelets that occur in one single class only
        # initialize a dictionary that stores lists of class-shapelets
        # for current dimension
        shapelets_class = {}
        # initialize a dictionary that stores class-shapelets
        # heatmaps for current dimension
        heat_maps = {}
        for c in np.unique(y_train):
            shapelets_class[c] = []
            heat_maps[c] = {}

        # keep shapelets that occur in one single class only
        for i, shapelet_classes in enumerate(shapelets_classes):
            for c in np.unique(y_train):
                if np.all(np.asarray(shapelet_classes) == c):
                    shapelets_class[c].append(i)
                if len(shapelets_classes) == 0:
                    print(
                        "All shapelets belong to more than one class exclusively\n Please consider using different parameters for the Shapelet Transform or the fit function"
                    )
                    return

        for c in np.unique(y_train):
            all_shapelets_class[c].append(shapelets_class[c])
            ###Get shapelet_locations distributions per exclusive class
            for s in shapelets_class[c]:
                heat_map = np.zeros(ts_length)
                num_occurences = 0
                for sl in all_shapelet_locations[dim][s]:
                    for idx in range(sl[1], sl[2]):
                        heat_map[idx] += 1
                    num_occurences += 1
                heat_map = heat_map / num_occurences
                heat_maps[c][s] = heat_map
            all_heat_maps[c].append(heat_maps[c])
    print("Shapelet by index per class and dimension:", all_shapelets_class)
    return (
        fitted_shapelets,
        threshold,
        all_heat_maps,
        all_shapelets_class,
    )


def sets_explain(
    instance_x,
    target,
    data,
    transformer,
    model,
    ts_length,
    st_shapelets,
    threshhold,
    all_shapelets_class,
    all_heat_maps,
    all_shapelets_scores,
    random_seed=42,
):
    random.seed(random_seed)

    X_train, y_train = data

    # get distance for timeseries to explain
    shapelets_distances_test = transformer.transform(
        from_3d_numpy_to_nested(np.expand_dims(instance_x, axis=0))
    )

    all_shapelet_locations_test, _ = get_all_shapelet_locations_scaled_threshold_test(
        [np.expand_dims(shapelets_distances_test, axis=0)],
        instance_x.shape[1],
        threshhold,
    )

    # Sort dimensions by their highest shapelet scores
    shapelets_best_scores = []
    for dim in range(len(st_shapelets)):
        shapelets_best_scores.append(max(all_shapelets_scores[dim]))
        shapelets_best_scores[dim] = np.argsort(shapelets_best_scores[dim])[::-1]

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

    orig_c = int(np.argmax(model.predict(to_tff(instance_x))))
    if len(target) > 1:
        target.remove(orig_c)
    for target_c in target:
        target_knn = knns[target_c]
        # starting the with the most important dimension, start CF generation
        for dim in range(len(shapelets_best_scores)):
            original_all_shapelets_class = all_shapelets_class[orig_c]
            all_target_heat_maps = all_heat_maps[target_c]
            target_knn = knns[target_c]

            nn_idx = get_nearest_neighbor(
                target_knn, instance_x, orig_c, X_train, y_train
            )
            original_all_shapelets_class = all_shapelets_class[orig_c][dim]
            all_target_heat_maps = all_heat_maps[target_c][dim]

            cf_dims = np.zeros((len(shapelets_best_scores), ts_length))

            cf = instance_x.copy()

            cf_pred = model.predict(to_tff(cf))
            cf_pred = np.argmax(cf_pred)
            if target_c != cf_pred:
                # Get the locations where the original class shapelets occur
                all_locs = get_shapelets_locations_test(
                    instance_x,
                    all_shapelet_locations_test,
                    dim,
                    original_all_shapelets_class,
                )
                # Replace the original class shapelets with nn values
                for c_i in all_locs:
                    for loc in all_locs.get(c_i):
                        cf_pred = model.predict(to_tff(cf))
                        cf_pred = np.argmax(cf_pred)
                        if target_c != cf_pred:
                            # print('Removing original shapelet')
                            nn = X_train[nn_idx].reshape(-1)

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
                for idx, target_shapelet_idx in enumerate(all_target_heat_maps.keys()):
                    cf_pred = model.predict(to_tff(cf))
                    cf_pred = np.argmax(cf_pred)
                    if target_c != cf_pred:
                        # print('Introducing new shapelet')
                        h_m = all_target_heat_maps[target_shapelet_idx]
                        center = (
                            np.argwhere(h_m > 0)[-1][0] - np.argwhere(h_m > 0)[0][0]
                        ) // 2 + np.argwhere(h_m > 0)[0][0]

                        target_shapelet = st_shapelets[dim][idx][0]
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
            cf_pred = model.predict(to_tff(cf))
            cf_pred = np.argmax(cf_pred)
            if target_c == cf_pred:
                return cf, cf_pred
            elif target_c != cf_pred:
                # Try all combinations of dimensions
                for L in range(0, len(shapelets_best_scores) + 1):
                    for subset in itertools.combinations(shapelets_best_scores, L):
                        if len(subset) >= 2:
                            cf = instance_x.copy()
                            for dim_ in subset:
                                cf[dim_] = cf_dims[dim_]
                            cf_pred = model.predict(to_tff(cf))
                            cf_pred = np.argmax(cf_pred)
                            if target_c == cf_pred:
                                break
            if target_c == cf_pred:
                return cf, cf_pred
            else:
                return None, None
