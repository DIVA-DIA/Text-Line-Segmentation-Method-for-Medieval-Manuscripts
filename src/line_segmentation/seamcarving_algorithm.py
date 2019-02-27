import logging
import sys
import time

import cv2
import numpy as np
import numba

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
    seam = []

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

        seam.append([i, previous])

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

            seam.append([i, previous])

    return seam


@numba.jit()
def draw_seams(img, seams):
    bidirectional = True
    # get the first seam and check if the x coordinate at position len/2 is width or not
    # because of integer we get first element of the right to left seam if there is one
    if len(seams[0]) == img.shape[1]:
        bidirectional = False
    for seam in seams:
        # Get the seam from the left [0] and the seam from the right[1]
        if bidirectional:
            split_seams = np.split(np.asarray(seam), 2)
            cv2.polylines(img, np.int32([np.asarray(split_seams[0])]), False, (0, 0, 0))
            cv2.polylines(img, np.int32([np.asarray(split_seams[1])]), False, (255, 255, 255))
        else:
            cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 0, 0))


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

        seam = horizontal_seam(energy_map, penalty_reduction=penalty_reduction, bidirectional=False)
        seams.append(seam)
    return seams