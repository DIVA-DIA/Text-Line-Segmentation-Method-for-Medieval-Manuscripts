import logging
import time

import numpy as np
from scipy.spatial import distance


def majority_voting(centroids, seams):
    """
    Splits the centroids into bins according to how many seams cross them
    """
    # -------------------------------
    start = time.time()
    # -------------------------------

    # strip seams of x coordinate, which is totally useless as the x coordinate is basically the index in the array
    seams = np.array([np.array(s)[:, 1] for s in seams])

    # for each centroid, compute how many seams are above it
    values = np.zeros([len(centroids)])
    for i, centroid in enumerate(centroids):
        cx = int(centroid[1])
        cy = int(centroid[0])
        for seam in seams:
            # if the seam is below the centroid at the centroid X position
            if seam[cx] > cy:
                values[i] = values[i] + 1

    small_bins = [42]
    while len(small_bins) > 0:
        # split values into bins index
        bin_index = np.digitize(values, np.unique(values))
        unique_bins , bin_size = np.unique(bin_index, return_counts=True)

        # look for outliers and merge them into bigger clusters
        if small_bins[0] == 42:
            avg = np.mean(bin_size)*0.25

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
                upper = distance.cdist(XA, centroids[np.where(bin_index == bin_index[loc_p])], 'euclidean').min()
                lower = distance.cdist(XA, centroids[np.where(bin_index == bin_index[loc_m])], 'euclidean').min()

                # -------------------------------------
                # COMMENTED but kept for legacy reasons
                # adapt index for prevent out of bounds
                # loc_p = loc + 1 if loc + 1 < len(values) else loc
                # loc_m = loc - 1 if loc > 0 else loc
                # compute distances to neighbors with the bin index
                # upper = abs(values[loc_p] - values[loc])
                # lower = abs(values[loc_m] - values[loc])

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