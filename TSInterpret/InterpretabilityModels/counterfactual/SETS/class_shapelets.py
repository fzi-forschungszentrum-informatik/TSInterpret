# Author: Omar Bahri

import os
import random
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils import (
    get_all_shapelet_locations_scaled_threshold,
    get_all_shapelet_locations_scaled_threshold_test,
)


def main():
    random.seed(42)

    dataset_name = sys.argv[1]
    time_contract_in_mins = int(sys.argv[2])
    max_perc = float(sys.argv[3])

    # name of current run (dataset + parameters combination)
    run_name = "_".join([dataset_name, str(time_contract_in_mins), str(max_perc)])

    # path of intermediary results directory
    inter_results = os.path.abspath(os.path.join("results", "util_data", run_name))

    X_train = np.load(
        os.path.abspath(os.path.join("data", dataset_name, "X_train.npy"))
    )
    y_train = np.load(
        os.path.abspath(os.path.join("data", dataset_name, "y_train.npy"))
    )
    X_test = np.load(os.path.abspath(os.path.join("data", dataset_name, "X_test.npy")))
    y_test = np.load(os.path.abspath(os.path.join("data", dataset_name, "y_test.npy")))

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    st_shapelets = np.load(
        os.path.join(inter_results, "shapelets.pkl"), allow_pickle=True
    )
    shapelets_distances = np.load(
        os.path.join(inter_results, "shapelets_distances.pkl"), allow_pickle=True
    )
    shapelets_distances_test = np.load(
        os.path.join(inter_results, "shapelets_distances_test.pkl"), allow_pickle=True
    )

    # length of time series in dataset
    ts_length = X_train.shape[2]

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
            del st_shapelets[dim][index]
            del all_shapelet_locations[dim][index]
            try:
                del all_shapelet_locations_test[dim][index]
            except Exception:
                pass
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
        for i, shapelet_classes in enumerate(shapelets_classes):
            for c in np.unique(y_train):
                if np.all(np.asarray(shapelet_classes) == c):
                    shapelets_class[c].append(i)

        for c in np.unique(y_train):
            all_shapelets_class[c].append(shapelets_class[c])

            print(all_shapelets_class)

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
            all_heat_maps[c].append(heat_maps[c])
            all_heat_maps[c].append(heat_maps[c])
            all_heat_maps[c].append(heat_maps[c])

    # save intermediate results
    np.save(os.path.join(inter_results, "all_heat_maps.npy"), all_heat_maps)
    print(all_shapelets_class)
    np.save(os.path.join(inter_results, "all_shapelets_class.npy"), all_shapelets_class)
    all_shapelet_locations = np.array(all_shapelet_locations, dtype=object)
    all_shapelet_locations_test = np.array(all_shapelet_locations_test, dtype=object)
    np.save(
        os.path.join(inter_results, "all_shapelet_locations.npy"),
        all_shapelet_locations,
    )
    np.save(
        os.path.join(inter_results, "all_shapelet_locations_trth_test.npy"),
        all_shapelet_locations_test,
    )


if __name__ == "__main__":
    main()
