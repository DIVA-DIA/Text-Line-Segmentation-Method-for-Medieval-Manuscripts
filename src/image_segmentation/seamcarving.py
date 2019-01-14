import sys

import cv2
import numpy as np
import numba

"""
    Code from: https://github.com/danasilver/seam-carving/blob/master/seamcarve.py
    homepage: http://www.faculty.idc.ac.il/arik/SCWeb/imret/
"""


@numba.jit()
def horizontal_seam(energies, penalty=True, penalty_div=3000):
    height, width = energies.shape[:2]
    # the y position we started
    ori_y = 0
    previous = 0
    seam = []

    # quadratic penalty
    # (ori_y - (privious - 1)) ** 2

    for i in range(0, width, 1):
        col = energies[:, i]
        if i == 0:
            ori_y = previous = np.argmin(col)
        else:
            top = col[previous - 1] if previous - 1 >= 0 else 1e6
            middle = col[previous]  # if previous != height else 1e6
            bottom = col[previous + 1] if previous + 1 < height else 1e6

            if penalty:
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


def build_seam_energy_map(energy_map):
    # TODO take min
    return calculate_forward_energy(energy_map) + calculate_backward_energy(energy_map)


# calculates the energy values from left to right
@numba.jit()
def calculate_forward_energy(energy_map):
    seam_map = np.copy(energy_map)
    height, width = energy_map.shape[:2]

    for i in range(1, width, 1):
        last_col = seam_map[:, i - 1]
        current_col = seam_map[:, i]

        for idx, cell in enumerate(current_col):
            top = last_col[idx - 1] if idx != 0 else sys.maxsize
            mid = last_col[idx]
            bot = last_col[idx + 1] if idx + 1 < height else sys.maxsize

            current_col[idx] = cell + np.min([top, mid, bot])

    return seam_map


# calculates the energy values from right to left
@numba.jit()
def calculate_backward_energy(energy_map):
    seam_map = np.copy(energy_map)
    height, width = energy_map.shape[:2]

    for i in range(width - 2, -1, -1):
        last_col = seam_map[:, i + 1]
        current_col = seam_map[:, i]

        for idx, cell in enumerate(current_col):
            top = last_col[idx - 1] if idx != 0 else sys.maxsize
            mid = last_col[idx]
            bot = last_col[idx + 1] if idx + 1 < height else sys.maxsize

            current_col[idx] = cell + np.min([top, mid, bot])

    return seam_map

# def remove_horizontal_seam(img, seam):
#     height, width, bands = img.shape
#     removed = np.zeros((height - 1, width, bands), np.uint8)
#
#     for x, y in reversed(seam):
#         removed[0:y, x] = img[0:y, x]
#         removed[y:height - 1, x] = img[y + 1:height, x]
#
#     return removed
#
#
# def remove_vertical_seam(img, seam):
#     height, width, bands = img.shape
#     removed = np.zeros((height, width - 1, bands), np.uint8)
#
#     for x, y in reversed(seam):
#         removed[y, 0:x] = img[y, 0:x]
#         removed[y, x:width - 1] = img[y, x + 1:width]
#
#     return removed
