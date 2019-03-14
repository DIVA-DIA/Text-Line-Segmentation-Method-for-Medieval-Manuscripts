import itertools
import logging
import os
import time

import cv2
import numpy as np

from src.line_segmentation.seamcarving_algorithm import draw_seams_red
from src.line_segmentation.utils.util import calculate_asymmetric_distance, save_img


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

    # for each centroid, compute how many seams are above it
    values = count_seams_below(centroids, seams)

    small_bins = [42]  # Just to enter the while loop once
    while len(small_bins) > 0:
        # split values into bins index
        bin_index, bin_size, unique_bins = split_into_bins_and_index(values)

        # look for outliers and merge them into bigger clusters
        if small_bins[0] == 42:
            avg = np.mean(bin_size[bin_size>1])*0.25

            # Compute average centroid horizontal distance:
            # distances = []
            # for bin in unique_bins:
            #     distances.extend(compute_avg_pairwise_distance(centroids[np.where(bin_index == bin)]))
            # threshold = 5 * np.mean(distances)
            #
            # # Scatter bins which have an anomaly in the avg distance
            # for bin in unique_bins:
            #     locs = np.where(bin_index == bin)
            #     if check_for_anomaly(centroids[locs], threshold):
            #         # Compute the offset for the scattered bin
            #         offset = np.array(range(0, len(locs[0])))
            #         # Assign them to the bin, thus scattering it into many single-centroid bins
            #         values[locs] -= offset
            #         # Get index of next bin
            #         nb = int(np.max(locs)) + 1
            #         # Adjust following bins accordingly
            #         values[nb:] -= np.max(offset)

        # Detect clusters which are too small
        small_bins = unique_bins[np.where(bin_size < avg)]

        # Merge small bins
        merge_small_bins(bin_index, centroids, small_bins, values)


    # Split the centroids into bins according to the clusters
    lines = []
    for bin in unique_bins:
        lines.append(list(centroids[np.where(bin_index == bin)]))

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return lines, centroids, values


def count_seams_below(centroids, seams):
    values = np.zeros([len(centroids)])
    for i, centroid in enumerate(centroids):
        cx = int(centroid[1])
        cy = int(centroid[0])
        for seam in seams:
            # if the seam is above the centroid at the centroid X position
            if seam[cx] > cy:
                values[i] = values[i] + 1
    return values


def merge_small_bins(bin_index, centroids, small_bins, values):
    for bin in small_bins:
        # find the indexes of the samples which are undernumbered
        for loc in np.where(bin_index == bin)[0]:
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

            upper = np.array([calculate_asymmetric_distance(XA, c, 1, 5) for c in
                              centroids[np.where(bin_index == bin_index[loc_p])]]).min()
            lower = np.array([calculate_asymmetric_distance(XA, c, 1, 5) for c in
                              centroids[np.where(bin_index == bin_index[loc_m])]]).min()

            values[loc] = values[loc_m] if (upper == 0 or upper > lower) and lower != 0 else values[loc_p]


def split_into_bins_and_index(values):
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


def draw_bins(img, centroids, root_output_path, seams, bins):
    # create white image
    binning_img = np.zeros(img.shape[0:2], dtype=np.uint8)
    binning_img.fill(255)
    # get text location
    locs = np.array(np.where(img[:, :, 0].reshape(-1) != 0))[0:2, :]
    binning_img = binning_img.flatten()
    binning_img[locs] = 211
    binning_img = binning_img.reshape(img.shape[0:2])
    binning_img = np.stack((binning_img,) * 3, axis=-1)
    # draw seams
    draw_seams_red(binning_img, seams)
    overlay_img = binning_img.copy()
    # draw the centroids on the seam energy map
    for centroid, value in zip(centroids, bins):
        cv2.circle(overlay_img, (int(centroid[1]), int(centroid[0])), 25, (0, 255, 0), -1)
        cv2.putText(binning_img, str(int(value)), (int(centroid[1]) - 16, int(centroid[0]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay_img, 0.4, binning_img, 0.6, 0, binning_img)
    save_img(binning_img, path=os.path.join(root_output_path, 'energy_map', 'energy_map_bin_expl.png'))
