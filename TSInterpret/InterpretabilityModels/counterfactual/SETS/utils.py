import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.base import clone


# Fit the transformer to all dimensions of the dataset
class MultivariateTransformer:
    def __init__(self, st):
        self.st = st
        self.sts = None

    def fit(self, X, y=None):
        self.n_dims = X.shape[1]
        self.sts = [clone(self.st) for _ in range(self.n_dims)]

        for i, transformer in enumerate(self.sts):
            transformer.fit(X.iloc[:, i].to_frame(), y)
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for i, transformer in enumerate(self.sts):
            try:
                X_transformed.append(transformer.transform(X.iloc[:, i].to_frame()))
            except RuntimeError:
                continue

        X_new = pd.concat(X_transformed, axis=1)
        return X_new


# get the list of shapelets of a transformer
def get_shapelets(transformer):
    all_shapelets = []
    for st in transformer.sts:
        c = 0
        dim_shapelets = []
        for shapelet in st.shapelets:
            dim_shapelets.append(shapelet.data)
            c += 1
        all_shapelets.append(dim_shapelets)

    return all_shapelets


# get the list of shapelet scores of a transformer
def get_scores(transformer):
    all_scores = []
    for st in transformer.sts:
        dim_scores = []
        for shapelet in st.shapelets:
            dim_scores.append(shapelet.info_gain)
        all_scores.append(dim_scores)

    return np.asarray(all_scores, dtype=object)


# get the distance of shapelets from each other shapelet in the MTS
def get_shapelets_distances(transformer):
    all_shapelets_distances = []
    for st in transformer.sts:
        shapelets_distances = []
        for shapelet in st.shapelets:
            shapelets_distances.append(shapelet.distances)
        all_shapelets_distances.append(shapelets_distances)
    return all_shapelets_distances


# Given the shapelet_locations and shapelet_distances of one single shapelet, removes
# the overlapping shapelet locations except the closest to the original
def remove_similar_locations(shapelet_locations, shapelet_distances):
    # List to keep indices to be discarded
    to_discard = []

    # Sort the shapelet_locations by sample index, then by start index
    shapelet_locations = shapelet_locations[
        np.lexsort((shapelet_locations[:, 1], shapelet_locations[:, 0]))
    ]

    # Variables to store the currently selected shapelet
    current_dist = np.inf
    current_idx = -1  # the sample index
    current_start = -1
    current_end = -1
    current_i = -1  # the index in the shapelet_locations array

    i = 0
    for shapelet in shapelet_locations:
        idx = shapelet[0]
        start = shapelet[1]
        end = shapelet[2]
        # Check if this location overlaps the selected shapelet
        if idx == current_idx and (not (start >= current_end or end <= current_start)):
            dist = shapelet_distances[idx][start]

            # If the distance of this shapelet is smaller than the distance of the currently
            # selected shapelet, discard the currently selected shapelet and select this one
            if dist < current_dist:
                to_discard.append(current_i)
                current_i = i
                current_dist = dist
                # Widen shapelet l
                current_start = np.minimum(current_start, start)
                current_end = np.maximum(current_end, end)
            # Else, discard this shapelet
            else:
                to_discard.append(i)
        # If it doesn't overlap it, select this one
        else:
            current_idx = shapelet[0]
            current_start = shapelet[1]
            current_end = shapelet[2]
            current_dist = shapelet_distances[idx][start]
            current_i = i
        i += 1

    return np.delete(shapelet_locations, to_discard, axis=0)


# Given the shapelet_distances matrix of a given shapelet, get the locations of
# the closest shapelets from the entire dataset
def get_shapelet_locations_scaled_threshold(shapelet_distances, ts_length, threshold):
    # Compute the length of the shapelet
    shapelet_length = ts_length - shapelet_distances.shape[1] + 1

    # Get the indices of the n closest shapelets to the original shapelet
    s_indices = []
    for i in range(shapelet_distances.shape[0]):
        for j in range(shapelet_distances.shape[1]):
            # Compare to the threshold, scaled to shapelet length
            if shapelet_distances[i][j] / shapelet_length <= threshold:
                s_indices.append(np.array([i, j]))

    if len(s_indices) > 0:
        s_indices = np.asarray(s_indices)

        # Create an array to store the locations of the closest n shapelets
        shapelet_locations = np.empty(
            (s_indices.shape[0], s_indices.shape[1] + 1), dtype=np.uint32
        )
        # Each shapelet is represented by (sample_index, start, end)
        for i in range(shapelet_locations.shape[0]):
            shapelet_locations[i] = np.append(
                s_indices[i], s_indices[i][1] + shapelet_length
            )

        # Remove overlapping shapelets and keep the closest one to th original shapelet
        shapelet_locations = remove_similar_locations(
            shapelet_locations, shapelet_distances
        )

        return shapelet_locations

    else:
        return np.array(np.array([[-1, -1, -1]]), dtype=np.uint32)


# Returns the threshold used to select shapelet occurences based on a given percentage
def get_occurences_threshold(shapelets_distances, ts_length, percentage):
    # print(shapelets_distances, ts_length, percentage)
    # List to hold all distances values
    sds = []
    # Append the scaled distances
    for dim in shapelets_distances:
        for shapelet_distances in dim:
            # Compute the length of the shapelet
            shapelet_length = ts_length - shapelet_distances.shape[1] + 1
            for instance in shapelet_distances:
                for distance in instance:
                    sds.append(distance / shapelet_length)

    # Sort the distances ascendingly
    sds.sort()

    # Number of shapelet occurences to keep (per shapelet)
    n = int(percentage * len(sds))

    # Return the threshold distance to select the shapelet occurences to keep
    return sds[n]


# Get the locations of the closest shapelets for each timeseries across the
# entire dataset based on a chosen percentage
def get_all_shapelet_locations_scaled_threshold(
    shapelets_distances, ts_length, percentage
):
    # Get the threshold to be used for selecting shapelet occurences
    threshold = get_occurences_threshold(shapelets_distances, ts_length, percentage)

    all_shapelet_locations = []
    all_no_occurences = []

    for dim in shapelets_distances:
        dim_shapelet_locations = []
        no_occurences = []
        for i, shapelet in enumerate(dim):
            sls = get_shapelet_locations_scaled_threshold(
                shapelet, ts_length, threshold
            )
            if sls[0][0] != 4294967295:
                dim_shapelet_locations.append(sls)
            else:
                no_occurences.append(i)
        all_shapelet_locations.append(dim_shapelet_locations)
        all_no_occurences.append(no_occurences)
    return all_shapelet_locations, all_no_occurences, threshold


# Get the locations of the closest shapelets for each timeseries across the
# entire dataset based on the training threshold
def get_all_shapelet_locations_scaled_threshold_test(
    shapelets_distances, ts_length, threshold
):
    all_shapelet_locations = []
    all_no_occurences = []

    for dim in shapelets_distances:
        dim_shapelet_locations = []
        no_occurences = []
        for i, shapelet in enumerate(dim):
            sls = get_shapelet_locations_scaled_threshold(
                shapelet, ts_length, threshold
            )
            if sls[0][0] != 4294967295:
                dim_shapelet_locations.append(sls)
            else:
                no_occurences.append(i)
        all_shapelet_locations.append(dim_shapelet_locations)
        all_no_occurences.append(no_occurences)

    return all_shapelet_locations, all_no_occurences


def get_shapelets_locations_test(idx, all_sls, dim, all_shapelets_class):
    all_locs = {}
    try:
        for i, s in enumerate([all_sls[dim][j] for j in all_shapelets_class[dim]]):
            i_locs = []
            for loc in s:
                if loc[0] == idx:
                    loc = (loc[1], loc[2])
                    i_locs.append(loc)
            all_locs[i] = i_locs
    except Exception as ex:
        pass
    return all_locs


##Optimize by fitting outside or returning a list of all nns at once
## Reworked so that only training data is available.
def get_nearest_neighbor(knn, instance_x, pred_label, x_train, y_train):
    # pred_label = y_pred[idx]
    target_labels = np.argwhere(y_train != pred_label)

    X_train_knn = instance_x.reshape(1, instance_x.shape[0], instance_x.shape[1])
    X_train_knn = np.swapaxes(X_train_knn, 1, 2)

    _, nn = knn.kneighbors(X_train_knn)
    # print("TARGETLABELS", [t[0] for t in target_labels], [int(nn[0][0])])
    nn_idx = None
    try:
        nn_idx = [t[0] for t in target_labels][int(nn[0][0])]
    except:
        pass
    return nn_idx  # None
