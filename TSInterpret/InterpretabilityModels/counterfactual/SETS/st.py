# Author: Omar Bahri

import os
import random
import sys

import numpy as np
from shapelets import ContractedShapeletTransform
from sklearn.preprocessing import LabelEncoder
from sktime_convert import from_3d_numpy_to_nested
from utils import MultivariateTransformer, save_shapelets_distances, save_transformer


def main():
    random.seed(42)

    dataset_name = sys.argv[1]


    # Load training set
    print(
        dataset_name,
        np.load(
            os.path.abspath(os.path.join("data", dataset_name, "X_train.npy"))
        ).shape,
    )
    X_train = np.load(
        os.path.abspath(os.path.join("data", dataset_name, "X_train.npy"))
    )
    y_train = np.load(
        os.path.abspath(os.path.join("data", dataset_name, "y_train.npy"))
    )
    X_test = np.load(os.path.abspath(os.path.join("data", dataset_name, "X_test.npy")))
    y_test = np.load(os.path.abspath(os.path.join("data", dataset_name, "y_test.npy")))

    X_train = from_3d_numpy_to_nested(X_train)
    y_train = np.asarray(y_train)
    X_test = from_3d_numpy_to_nested(X_test)
    y_test = np.asarray(y_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # How long (in minutes) to extract shapelets for.
    # This is a simple lower-bound initially;
    # once time is up, no further shapelets will be assessed
    time_contract_in_mins = int(sys.argv[2])

    time_contract_in_mins_per_dim = int(time_contract_in_mins / X_train.shape[1])

    # Set lengths of shapelets to mine
    max_perc = float(sys.argv[3])
    print("SHAPE", X_train.shape)
    min_length, max_length = 3, 20  # int(max_perc / 100 * X_train.shape[1])

    # If the time contract per dimension is less than one minute, sample
    # time_contract_in_mins random dimensions and apply ST to them
    seed = 10

    if time_contract_in_mins_per_dim < 1:
        random.seed(seed)
        dims = [
            random.randint(0, X_train.shape[1] - 1)
            for p in range(0, int(time_contract_in_mins))
        ]

        X_train = X_train.iloc[:, dims]

        # Spend one minute on each dimension
        time_contract_in_mins_per_dim = 1

    # The initial number of shapelet candidates to assess per training series.
    # If all series are visited and time remains on the contract then another
    # pass of the data will occur
    initial_num_shapelets_per_case = 10

    # Whether or not to print on-going information about shapelet extraction.
    # Useful for demo/debugging
    verbose = 2

    st = ContractedShapeletTransform(
        time_contract_in_mins=time_contract_in_mins_per_dim,
        num_candidates_to_sample_per_case=initial_num_shapelets_per_case,
        min_shapelet_length=min_length,
        max_shapelet_length=max_length,
        verbose=verbose,
        predefined_ig_rejection_level=0.001,
        max_shapelets_to_store_per_class=30,
    )

    print("MIN", min_length, "MAX", max_length)

    transformer = MultivariateTransformer(st)

    transformer.fit(X_train, y_train)

    X_new = transformer.transform(X_train)

    # name of current run (dataset + parameters combination)
    run_name = "_".join([dataset_name, str(time_contract_in_mins), str(max_perc)])

    # path of intermediary results directory
    inter_results = os.path.abspath(os.path.join("results", "util_data", run_name))

    if not os.path.exists(inter_results):
        os.makedirs(inter_results)

    save_shapelets_distances(inter_results, transformer, test=False)
    np.save(os.path.join(inter_results, "X_new.npy"), X_new)
    save_transformer(inter_results, transformer)

    X_test_new = transformer.transform(X_test)

    np.save(os.path.join(inter_results, "X_test_new.npy"), X_test_new)
    save_shapelets_distances(inter_results, transformer, test=True)


if __name__ == "__main__":
    main()
