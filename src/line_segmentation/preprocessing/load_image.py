import errno

import cv2
import logging
import os
import time

import numpy as np


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    return cv2.imread(path)


def prepare_image(img, testing, cropping=True):
    # -------------------------------
    start = time.time()
    # -------------------------------

    if testing:
        img[:, :, 0] = 0
        img[:, :, 2] = 0
        locations = np.where(img == 127)
        img[:, :, 1] = 0
        img[locations[0], locations[1]] = 255
        if cropping:
            locs = np.array(np.where(img == 255))[0:2, ]
            img = img[np.min(locs[0, :]):np.max(locs[0, :]), np.min(locs[1, :]):np.max(locs[1, :])]

    else:
        # Erase green just in case
        img[:, :, 1] = 0
        # Find and remove boundaries (set to bg)
        locations = np.where(img == 128)
        img[locations[0], locations[1]] = 0
        # Find regular text and text + decoration
        locations_text = np.where(img == 8)
        locations_text_comment = np.where(img == 12)
        # Wipe the image
        img[:, :, :] = 0
        # Set the text to be white
        img[locations_text[0], locations_text[1]] = 255
        img[locations_text_comment[0], locations_text_comment[1]] = 255

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return img