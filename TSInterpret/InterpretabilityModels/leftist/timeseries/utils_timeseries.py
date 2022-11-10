import numpy as np


def start_stop(sequence, value, len_thresh=2):
    """
    Retrieve the interval of a consecutive value in a sequence superior to the specified length.

    Parameters:
        sequence (np.ndarray): the sequence of values.
        value (int): value to retrieve.
        len_tresh (int): minimum length of the interval to find.

    Returns
        intervals (np.ndarray): list of pair, index of the limit of intervals.
    """
    # "Enclose" mask with sentients to catch shifts later on
    mask = np.r_[False, np.equal(sequence, value), False]

    # Get the shifting indices
    idx = np.flatnonzero(mask[1:] != mask[:-1])

    # Get lengths
    lens = idx[1::2] - idx[::2]

    return idx.reshape(-1, 2)[lens > len_thresh] - [0, 1]
