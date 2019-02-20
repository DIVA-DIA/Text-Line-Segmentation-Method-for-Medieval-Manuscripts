import logging
import sys
import time

import cv2
import numpy as np
import numba

"""
    Code from: https://github.com/danasilver/seam-carving/blob/master/seamcarve.py
    homepage: http://www.faculty.idc.ac.il/arik/SCWeb/imret/
"""


@numba.jit()
def horizontal_seam(energies, penalty_div, bidirectional=False):
    """
    Spawns seams from the left to the right or from both directions. It returns the list of seams as point list.

    :param energies: the energy map
    :param penalty_div: if the penalty is smaller or equal to 0 we wont apply a penalty reduction
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

            if penalty_div > 0:
                top += ((ori_y - (previous - 1)) ** 2) / penalty_div
                middle += ((ori_y - previous) ** 2) / penalty_div
                bottom += + ((ori_y - (previous + 1)) ** 2) / penalty_div

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

                if penalty_div > 0:
                    top += ((ori_y - (previous - 1)) ** 2) / penalty_div
                    middle += ((ori_y - previous) ** 2) / penalty_div
                    bottom += + ((ori_y - (previous + 1)) ** 2) / penalty_div

                previous = previous + np.argmin([top, middle, bottom]) - 1

            seam.append([i, previous])

    return seam


@numba.jit()
def draw_seam(img, seam, show=False):
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 0, 0))
    if show:
        cv2.imshow('seam', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

