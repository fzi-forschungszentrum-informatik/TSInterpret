import random
import warnings

import numpy as np
import pandas as pd
from deap import creator, tools
from pyts.utils import windowed_view
from scipy.fft import irfft, rfft

warnings.filterwarnings("ignore", category=DeprecationWarning)


def eval(x, mop, return_values_of):
    """
    Help Function.
    Args:
            x (np.array): instance to evaluate.
            mop (pymop.Problem): instance of Multiobjective Problem.
            return_values_of (np.array): Specify array to return.

    Returns:
            [np.array]: fitnessvalues
    """
    # print('eval')
    return mop.evaluate([x], return_values_of)  # , mop.prediction


def evaluate_pop(pop, toolbox):
    for ind in pop:
        out = toolbox.evaluate(ind)
        # print(out)
        if type(out) == tuple:
            ind.fitness.values = out
        else:
            ind.fitness.values = tuple(out[0])
        # print('IND Fitness',ind.fitness.values)
    return pop


def recombine(ind1, ind2):
    """Crossover"""

    window_size1 = ind1.window
    window_size2 = ind2.window

    shape = np.array(ind1).shape[1]

    num_channels = len(ind1.channels)
    channel1 = ind1.channels
    mutation = ind1.mutation

    if window_size1 == 1:
        ind1, ind2 = tools.cxUniform(
            np.array(ind1).reshape(num_channels, shape),
            np.array(ind2).reshape(num_channels, shape),
            indpb=0.1,
        )
    else:

        if (shape / window_size1).is_integer():

            ind1 = windowed_view(
                np.array(ind1).reshape(num_channels, shape),
                window_size1,
                window_step=window_size1,
            )

            ind2 = windowed_view(
                np.array(ind2).reshape(num_channels, shape),
                window_size1,
                window_step=window_size1,
            )

        else:
            # print('CX else')
            shape_new = window_size1 * (int(shape / window_size1) + 1)
            padded = np.zeros((num_channels, shape_new))
            padded2 = np.zeros((num_channels, shape_new))
            padded[:, :shape] = np.array(ind1).reshape(num_channels, shape)
            padded2[:, :shape] = np.array(ind2).reshape(num_channels, shape)
            ind1 = windowed_view(
                np.array(padded).reshape(num_channels, shape_new),
                window_size1,
                window_step=window_size1,
            )
            ind2 = windowed_view(
                np.array(padded2).reshape(num_channels, shape_new),
                window_size1,
                window_step=window_size1,
            )

        if num_channels == 1:
            ind1[0], ind2[0] = tools.cxUniform(
                np.array(ind1[0]).tolist(), np.array(ind2[0]).tolist(), indpb=0.1
            )

        else:
            items = np.where(channel1 == 1)
            if len(items[0]) != 0:
                for item in items:
                    ind1[item], ind2[item] = tools.cxUniform(
                        np.array(ind1[item]).tolist(),
                        np.array(ind2[item]).tolist(),
                        indpb=0.1,
                    )

    shape_new = np.array(ind1).reshape(1, -1).shape[1]
    if shape_new > shape:

        diff = shape_new - shape
        ind1 = np.array(ind1).reshape(num_channels, -1)[:, 0 : shape_new - diff]
        ind2 = np.array(ind2).reshape(num_channels, -1)[:, 0 : shape_new - diff]
    ind1 = creator.Individual(np.array(ind1).reshape(num_channels, -1).tolist())
    ind2 = creator.Individual(np.array(ind2).reshape(num_channels, -1).tolist())
    ind1.window = window_size1
    ind2.window = window_size2
    ind1.mutation = mutation
    ind2.mutation = mutation

    ind1.channels = channel1
    ind2.channels = channel1

    return ind1, ind2


def mutate(individual, means, sigmas, indpb, uopb):
    """Gaussian Mutation"""

    window = individual.window
    channels = individual.channels
    items = np.where(channels == 1)

    if len(items[0]) != 0:
        channel = random.choice(items[0])
        means = means[channel]
        sigmas = sigmas[channel]
        for i, m, s in zip(range(len(individual[int(channel)])), means, sigmas):

            if random.random() < indpb:
                individual[channel][i] = random.gauss(m, s)

    window, channels = mutate_hyperperameter(
        individual, window, channels, len(channels)
    )
    ind = creator.Individual(individual)
    ind.window = window
    ind.channel = channels
    ind.mutation = "mean"
    return (ind,)


def create_mstats():
    """Logging the Stats"""
    stats_y_distance = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_x_distance = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_changed_features = tools.Statistics(lambda ind: ind.fitness.values[2])
    mstats = tools.MultiStatistics(
        stats_y_distance=stats_y_distance,
        stats_x_distance=stats_x_distance,
        stats_changed_features=stats_changed_features,
    )
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats


def create_logbook():
    logbook = tools.Logbook()
    logbook.header = (
        "gen",
        "pop",
        "evals",
        "stats_y_distance",
        "stats_x_distance",
        "stats_changed_features",
    )
    # logbook.chapters["fitness"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_y_distance"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_x_distance"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_changed_features"].header = "std", "min", "avg", "max"
    return logbook


def pareto_eq(ind1, ind2):
    """Determines whether two individuals are equal on the Pareto front
    Parameters
    ----------
    ind1: DEAP individual from the GP population
        First individual to compare
    ind2: DEAP individual from the GP population
        Second individual to compare
    Returns
    ----------
    individuals_equal: bool
        Boolean indicating whether the two individuals are equal on
        the Pareto front
    """
    return np.all(ind1.fitness.values == ind2.fitness.values)


def authentic_opposing_information(ind1, reference_set):

    window = ind1.window
    # window=10
    num_channels = len(ind1.channels)
    channels = ind1.channels
    shape = np.array(ind1).shape[-1]
    sample_series = random.choice(reference_set)
    if (shape / window).is_integer():
        ind1 = windowed_view(
            np.array(ind1).reshape(num_channels, shape), window, window_step=window
        )  # [0]
        sample_series = windowed_view(
            sample_series.reshape(num_channels, shape), window, window_step=window
        )  # [0]

    else:

        shape_new = window * (int(shape / window) + 1)
        padded = np.zeros((num_channels, shape_new))
        sample_padded = np.zeros((num_channels, shape_new))
        padded[:, :shape] = np.array(ind1).reshape(num_channels, shape)
        sample_padded[:, :shape] = sample_series.reshape(num_channels, shape)
        ind1 = windowed_view(
            np.array(padded).reshape(num_channels, shape_new),
            window,
            window_step=window,
        )
        sample_series = windowed_view(sample_padded, window, window_step=window)

    items = np.where(channels == 1)
    if len(items[0]) != 0:
        channel = random.choice(items[0])
        index = random.randint(0, len(ind1[0]) - 1)
        ind1[channel, index] = sample_series[channel, index]

    new_shape = ind1.reshape(num_channels, -1).shape[1]
    if new_shape > shape:

        diff = shape_new - shape
        ind1 = np.array(ind1).reshape(num_channels, -1)[:, 0 : shape_new - diff]

    ind1 = ind1.reshape(num_channels, -1)

    ind1 = creator.Individual(np.array(ind1).reshape(num_channels, -1).tolist())

    window, channels = mutate_hyperperameter(ind1, window, channels, num_channels)
    ind1.window = window
    ind1.channels = channels
    ind1.mutation = "auth"
    return (ind1,)


def frequency_band_mapping(ind1, reference_set):
    num_channels = len(ind1.channels)
    channels = ind1.channels
    window = ind1.window
    ind1 = np.array(ind1).reshape(1, -1, reference_set.shape[-1])
    shape = ind1.shape
    fourier_timeseries = rfft(ind1)  # Fourier transformation of timeseries

    fourier_reference_set = rfft(
        np.array(reference_set)
    )  # Fourier transformation reference set
    len_fourier = fourier_timeseries.shape[-1]  # lentgh of fourier

    # Define variables
    length = 1
    num_slices = 1

    # Set up dataframe for slices with start and end value
    slices_start_end_value = pd.DataFrame(columns=["Slice_number", "Start", "End"])
    # Include the first fourier band which should not be perturbed
    new_row = {"Slice_number": 0, "Start": 0, "End": 1}
    # append row to the dataframe
    slices_start_end_value = slices_start_end_value.append(new_row, ignore_index=True)
    start_idx = length
    end_idx = length
    while length < len_fourier:
        start_idx = length  # Start value
        end_idx = start_idx + num_slices**2  # End value
        end_idx = min(end_idx, len_fourier)

        new_row = {"Slice_number": num_slices, "Start": start_idx, "End": end_idx}
        # append row to the dataframe
        slices_start_end_value = slices_start_end_value.append(
            new_row, ignore_index=True
        )

        length = length + end_idx - start_idx
        num_slices = num_slices + 1

    feature = np.where(channels == 1)
    if len(feature[0]) != 0:
        # Select Feature to be changed
        num_feature = random.choice(feature[0])

        tmp_fourier_series = np.array(fourier_timeseries.copy())
        max_row_idx = fourier_reference_set.shape[0]
        rand_idx = np.random.randint(0, max_row_idx)
        idx = random.randint(0, len(slices_start_end_value) - 1)
        start_idx = slices_start_end_value["Start"][idx]
        end_idx = slices_start_end_value["End"][idx]
        tmp_fourier_series[0, num_feature, start_idx:end_idx] = fourier_reference_set[
            rand_idx, num_feature, start_idx:end_idx
        ].copy()
        perturbed_fourier_retransform = irfft(tmp_fourier_series, n=shape[2])
        ind1 = creator.Individual(
            np.array(perturbed_fourier_retransform).reshape(shape[1], shape[2]).tolist()
        )
    else:
        ind1 = creator.Individual(np.array(ind1).reshape(shape[1], shape[2]).tolist())
    window, channels = mutate_hyperperameter(ind1, window, channels, num_channels)
    ind1.channels = channels
    ind1.window = window
    ind1.mutation = "freq"
    return (ind1,)


def mutate_mean(ind1, reference_set):
    window = ind1.window
    num_channels = len(ind1.channels)
    channels = ind1.channels
    means = reference_set.mean(axis=0)
    sigmas = reference_set.std(axis=0)
    (ind1,) = mutate(ind1, means=means, sigmas=sigmas, indpb=0.56, uopb=0.32)
    ind1.mutation = "mean"
    window, channels = mutate_hyperperameter(ind1, window, channels, num_channels)
    ind1.channels = channels
    ind1.window = window
    return (ind1,)


def mutate_both(ind1, reference_set):
    """Still TODO"""
    if ind1.mutation == "auth":
        (ind1,) = authentic_opposing_information(ind1, reference_set)
    elif ind1.mutation == "freq":
        (ind1,) = frequency_band_mapping(ind1, reference_set)
    if ind1.mutation == "mean":
        means = reference_set.mean(axis=0)  #
        sigmas = reference_set.std(axis=0)
        (ind1,) = mutate(ind1, means=means, sigmas=sigmas, indpb=0.56, uopb=0.32)
    return (ind1,)


def mutate_hyperperameter(ind1, window, channels, num_channels):
    window = window
    channels = channels
    if random.random() < 0.5:
        window = random.randint(1, np.floor(0.5 * np.array(ind1).shape[-1]))
    return window, channels
