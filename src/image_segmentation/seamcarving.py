import cv2
import numpy as np
import numba

"""
    Code from: https://github.com/danasilver/seam-carving/blob/master/seamcarve.py
    homepage: http://www.faculty.idc.ac.il/arik/SCWeb/imret/
"""


@numba.jit()
def horizontal_seam(energies):
    height, width = energies.shape[:2]
    previous = 0
    seam = []

    for i in range(0, width, 1):
        col = energies[:, i]

        if i == 0:
            previous = np.argmin(col)
        else:
            top = col[previous - 1] if previous - 1 >= 0 else 1e6
            middle = col[previous]
            bottom = col[previous + 1] if previous + 1 < height else 1e6

            previous = previous + np.argmin([top, middle, bottom])

        seam.append([i, previous])

    return seam


def vertical_seam(energies):
    height, width = energies.shape[:2]
    previous = 0
    seam = []

    for i in range(height - 1, -1, -1):
        row = energies[i, :]

        if i == height - 1:
            previous = np.argmin(row)
            seam.append([previous, i])
        else:
            left = row[previous - 1] if previous - 1 >= 0 else 1e6
            middle = row[previous]
            right = row[previous + 1] if previous + 1 < width else 1e6

            previous = previous + np.argmin([left, middle, right]) - 1
            seam.append([previous, i])

    return seam


def draw_seam(img, seam, show=False):
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 0))
    if show:
        cv2.imshow('seam', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
