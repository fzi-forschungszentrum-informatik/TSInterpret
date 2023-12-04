# Author: Omar Bahri

import os
import random
import sys

import numpy as np

from TSInterpret.InterpretabilityModels.counterfactual.SETS.utils import (
    get_all_shapelet_locations_scaled_threshold,
    get_all_shapelet_locations_scaled_threshold_test,
)


def get_class_shapelets(
    data,
    ts_length,
    st_shapelets,
    shapelets_distances,
    shapelets_distances_test,
):
    random.seed(42)

    X_train, y_train, X_test, y_test = data

    # the percentage of shapelet occurrences to keep
    occ_threshold = 1e-1

    # get the shapelets locations in the training and testing sets
    (
        all_shapelet_locations,
        all_no_occurences,
        threshold,
    ) = get_all_shapelet_locations_scaled_threshold(
        shapelets_distances, ts_length, occ_threshold / 100.0
    )

    all_shapelet_locations_test, _ = get_all_shapelet_locations_scaled_threshold_test(
        shapelets_distances_test, ts_length, threshold
    )

    del shapelets_distances

    # initialize a dictionary that stores lists of class-shapelets
    all_shapelets_class = {}
    # initialize a dictionary that stores lists of class-shapelets heatmaps
    all_heat_maps = {}
    for c in np.unique(y_train):
        all_shapelets_class[c] = []
        all_heat_maps[c] = []

    # get the shapelet classes and their heatmaps at each dimension
    for dim in range(X_test.shape[1]):
        for index in sorted(all_no_occurences[dim], reverse=True):
            del st_shapelets[dim][index]

        # Get shapelets class occurences
        shapelets_classes = []
        for shapelet_locations in all_shapelet_locations[dim]:
            shapelet_classes = []
            for sl in shapelet_locations:
                shapelet_classes.append(y_train[sl[0]])
            shapelets_classes.append(shapelet_classes)

        not_one_class = []

        # Find shapelets that happen exclusively under one class
        for i, shapelet_class in enumerate(shapelets_classes):
            if len(np.unique(shapelet_class)) > 1:
                not_one_class.append(i)

        for index in sorted(not_one_class, reverse=True):
            try:
                del st_shapelets[dim][index]
                del all_shapelet_locations[dim][index]
                del all_shapelet_locations_test[dim][index]
                del shapelets_classes[index]
            except:
                pass

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

    return (
        all_heat_maps,
        all_shapelets_class,
        np.array(all_shapelet_locations, dtype=object),
        np.array(all_shapelet_locations_test, dtype=object),
    )
