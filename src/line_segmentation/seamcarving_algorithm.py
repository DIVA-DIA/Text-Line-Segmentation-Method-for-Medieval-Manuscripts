import itertools
import sys

import cv2
import numba
import numpy as np

import src.line_segmentation

"""
    Code from: https://github.com/danasilver/seam-carving/blob/master/seamcarve.py
    homepage: http://www.faculty.idc.ac.il/arik/SCWeb/imret/
"""


@numba.jit()
def horizontal_seam(energies, penalty_reduction, bidirectional=False):
    """
    Spawns seams from the left to the right or from both directions. It returns the list of seams as point list.

    :param energies: the energy map
    :param penalty_reduction: if the penalty_reduction is smaller or equal to 0 we wont apply a penalty reduction
    :param bidirectional: if True there will be seams from left to right and right to left, else just from left to right
    :return: seams as point list
    """
    height, width = energies.shape[:2]
    # the y position we started (needed for the penalty)
    ori_y = 0
    # the last point we visit
    previous = 0
    # the points of the seam
    # from left to right
    seam_forward = []
    # from right to left
    seam_backward = []

    # spawns seams from left to right
    for i in range(0, width, 1):
        col = energies[:, i]
        if i == 0:
            ori_y = previous = np.argmin(col)
        else:
            top = col[previous - 1] if previous - 1 >= 0 else sys.maxsize
            middle = col[previous]
            bottom = col[previous + 1] if previous + 1 < height else sys.maxsize

            if penalty_reduction > 0:
                top += ((ori_y - (previous - 1)) ** 2) / penalty_reduction
                middle += ((ori_y - previous) ** 2) / penalty_reduction
                bottom += + ((ori_y - (previous + 1)) ** 2) / penalty_reduction

            previous = previous + np.argmin([top, middle, bottom]) - 1

        seam_forward.append([i, previous])

    # spawns seams from right to left
    if bidirectional:
        for i in range(width-1, -1, -1):
            col = energies[:, i]
            if i == width-1:
                ori_y = previous = np.argmin(col)
            else:
                top = col[previous - 1] if previous - 1 >= 0 else sys.maxsize
                middle = col[previous]
                bottom = col[previous + 1] if previous + 1 < height else sys.maxsize

                if penalty_reduction > 0:
                    top += ((ori_y - (previous - 1)) ** 2) / penalty_reduction
                    middle += ((ori_y - previous) ** 2) / penalty_reduction
                    bottom += + ((ori_y - (previous + 1)) ** 2) / penalty_reduction

                previous = previous + np.argmin([top, middle, bottom]) - 1

            seam_backward.append([i, previous])

    return [seam_forward, seam_backward[::-1]]


def draw_seams(img, seams, bidirectional=True):

    x_axis = np.expand_dims(np.array(range(0, len(seams[0]))), -1)
    seams = [np.concatenate((x, np.expand_dims(seam, -1)), axis=1) for seam, x in zip(seams, itertools.repeat(x_axis))]

    for i, seam in enumerate(seams):
        # Get the seam from the left [0] and the seam from the right[1]
        if bidirectional and i % 2 == 0:
            cv2.polylines(img, np.int32([seam]), False, (0, 0, 0), 3)  # Black
        else:
            cv2.polylines(img, np.int32([seam]), False, (255, 255, 255), 3)  # White


def draw_seams_red(img, seams, bidirectional=True):

    x_axis = np.expand_dims(np.array(range(0, len(seams[0]))), -1)
    seams = [np.concatenate((x, np.expand_dims(seam, -1)), axis=1) for seam, x in zip(seams, itertools.repeat(x_axis))]

    for i, seam in enumerate(seams):
        # Get the seam from the left [0] and the seam from the right[1]
            cv2.polylines(img, np.int32([seam]), False, (0, 0, 255), 3)  # Red


def get_seams(ori_energy_map, penalty_reduction, seam_every_x_pxl):
    # list with all seams
    seams = []
    # left most column of the energy map
    left_column_energy_map = np.copy(ori_energy_map[:, 0])
    # right most column of the energy map
    right_column_energy_map = np.copy(ori_energy_map[:, -1])
    # show_img(ori_enegery_map)
    for seam_at in range(0, ori_energy_map.shape[0], seam_every_x_pxl):
        energy_map = src.line_segmentation.preprocessing.energy_map.prepare_energy(ori_energy_map,
                                                                                   left_column_energy_map,
                                                                                   right_column_energy_map, seam_at)

        seams.extend(horizontal_seam(energy_map, penalty_reduction=penalty_reduction, bidirectional=True))

    # strip seams of x coordinate, which is totally useless as the x coordinate is basically the index in the array
    seams = np.array([np.array(s)[:, 1] for s in seams])

    return seams


def post_process_seams(energy_map, seams):
    # Check that the seams are as wide as the image
    assert energy_map.shape[1] == len(seams[0])

    # TODO implement a tabu-list to prevent two seams to repeatedly swap a third seam between them
    SAFETY_STOP = 100
    iteration = 0
    repeat = True
    while repeat:

        # Safety exit in case of endless loop meeting condition. See above.
        iteration += 1
        if iteration >= SAFETY_STOP:
            break

        repeat = False
        for index, seam_A in enumerate(seams):
            for seam_B in seams[index:]:
                # Compute seams overlap
                overlap = seam_A - seam_B

                # Smooth the overlap
                overlap[abs(overlap) < 10] = 0

                # Make the two seams really overlap
                seam_A[overlap == 0] = seam_B[overlap == 0]

                # Find non-zero sequences
                sequences = non_zero_runs(overlap)

                if len(sequences) > 0:
                    for i, sequence in enumerate(sequences):

                        target = sequence[1] - sequence[0]

                        left = sequence[0] - sequences[i - 1, 1] if i > 0 else sequence[0]
                        right = sequences[i + 1, 0] - sequence[1] if i < len(sequences)-1 else energy_map.shape[1] - sequence[1]

                        if target > left and target > right:
                            continue

                        repeat = True

                        # Expand the sequence into a range
                        sequence = range(*sequence)
                        # Compute the seam
                        energy_A = measure_energy(energy_map, seam_A, sequence)
                        energy_B = measure_energy(energy_map, seam_B, sequence)

                        # Remove the weaker seam sequence
                        if energy_A > energy_B:
                            seam_A[sequence] = seam_B[sequence]
                        else:
                            seam_B[sequence] = seam_A[sequence]

    return seams


def non_zero_runs(a):
    """
    Finding the consecutive non-zeros in a numpy array. Modified from:
    https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    """
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([1], np.equal(a, 0).view(np.int8), [1]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def measure_energy(energy_map, seam, sequence):
    """
    Compute the energy of that seams for the specified range
    """
    return energy_map[seam[sequence], sequence].sum()
