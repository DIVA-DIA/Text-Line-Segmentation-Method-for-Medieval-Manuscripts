import itertools
import logging
import time

import numpy as np
from scipy.spatial import distance

from src.line_segmentation.utils.util import calculate_asymmetric_distance


def majority_voting(connected_components, seams):
    """
    Splits the centroids into bins according to how many seams cross them
    """
    # -------------------------------
    start = time.time()
    # -------------------------------

    # Get the centroids and sort them
    centroids = np.asarray([cc.centroid[0:2] for cc in connected_components[1]])
    centroids = centroids[np.argsort(centroids[:, 0]), :]

    # strip seams of x coordinate, which is totally useless as the x coordinate is basically the index in the array
    #seams = np.array([np.array(s)[:, 1] for s in seams])

    # for each centroid, compute how many seams are above it
    values = np.zeros([len(centroids)])
    for i, centroid in enumerate(centroids):
        cx = int(centroid[1])
        cy = int(centroid[0])
        for seam in seams:
            # if the seam is above the centroid at the centroid X position
            if seam[cx] > cy:
                values[i] = values[i] + 1

    small_bins = [42]  # Just to enter the while loop once
    while len(small_bins) > 0:
        # split values into bins index
        bin_index, bin_size, unique_bins = split_into_bins_and_index(values, centroids)

        # look for outliers and merge them into bigger clusters
        if small_bins[0] == 42:
            avg = np.mean(bin_size[bin_size>1])*0.25

        # cluster which are too small
        small_bins = unique_bins[np.where(bin_size < avg)]

        for bin in small_bins:
            # find the indexes of the samples which are undernumbered
            for loc in np.where(bin_index==bin)[0]:
                # look for the next available bin below
                loc_p = loc + 1 if loc + 1 < len(values) else loc
                while bin_index[loc_p] == bin_index[loc]:
                    if loc_p + 1 < len(values):
                        loc_p += 1
                    else:
                        break

                # look for the next available bin above
                loc_m = loc - 1 if loc > 0 else loc
                while bin_index[loc_m] == bin_index[loc]:
                    if loc_m > 0:
                        loc_m -= 1
                    else:
                        break

                # compute distances to neighbors with the EUC distance
                XA = np.expand_dims(centroids[loc], axis=0)

                upper = np.array([calculate_asymmetric_distance(XA, c, 1, 5) for c in centroids[np.where(bin_index == bin_index[loc_p])]]).min()
                lower = np.array([calculate_asymmetric_distance(XA, c, 1, 5) for c in centroids[np.where(bin_index == bin_index[loc_m])]]).min()

                values[loc] = values[loc_m] if (upper == 0 or upper > lower) and lower != 0 else values[loc_p]


    # split the centroids into bins according to the clusters
    lines = []
    for bin in unique_bins:
        lines.append(list(centroids[np.where(bin_index == bin)]))

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return lines


def split_into_bins_and_index(values, centroids):
    # Compute the initial split
    bin_index, bin_size, unique_bins = binning(values)

    # Compute average centroid horizontal distance
    distances = []
    for bin in unique_bins:
        distances.extend(compute_avg_pairwise_distance(centroids[np.where(bin_index == bin)]))

    # Scatter bins which have an anomaly in the avg distance
    for bin in unique_bins:
        locs = np.where(bin_index == bin)
        if check_for_anomaly(centroids[locs], 4*np.std(distances)):
            # Get a set of unique numbers out of range of what contained in 'values' already
            u = int(np.max(values)) + 1
            scattered_indexes = np.array(range(u, u + len(locs[0])))
            # Assign them to the bin, thus scattering it into many single-centroid bins
            values[locs] = scattered_indexes

    # Update the value with the new bins
    bin_index, bin_size, unique_bins = binning(values)

    return bin_index, bin_size, unique_bins


def binning(values):
    # Bin index is a an array with the bin number for each entry in values
    bin_index = np.digitize(values, np.unique(values))
    # Get the bins values and their size
    unique_bins, bin_size = np.unique(bin_index, return_counts=True)
    return bin_index, bin_size, unique_bins


def compute_avg_pairwise_distance(centroids):
    # Sort the centroids based on their x-coordinate
    centroids = centroids[np.argsort(centroids[:, 1]), :]
    dist = []
    for c1, c2 in pairwise(centroids):
        # Compute the distance in the horizontal axis
        dist.append(c2[1] - c1[1])
    return dist


def check_for_anomaly(centroids, threshold):
    # Sort the centroids based on their x-coordinate
    centroids = centroids[np.argsort(centroids[:, 1]), :]
    for c1, c2 in pairwise(centroids):
        # Compute the distance in the horizontal axis; If it is higher thant a threshold, trigger
        if c2[1] - c1[1] > threshold:
            return True
    return False


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)