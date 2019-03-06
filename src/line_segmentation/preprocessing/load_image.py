import errno

import cv2
import logging
import os
import time

import numpy as np

from PIL import Image


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    if os.path.splitext(path)[1] == '.tif':
        img = np.asarray(Image.open(path), dtype=int)
        img[np.where(img == 0)] = 8
        img[np.where(img == 1)] = 0
        img = np.stack((img,)*3, axis=-1)
    else:
        img = cv2.imread(path)

    if img is None:
        raise Exception("Image is empty or corrupted", path)

    return img

def prepare_image(img, testing, cropping=True, vertical=False):
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
        ##################################################################
        # Protect against malformed inputs

        # Green channel should be empty
        assert len(np.unique(img[:, :, 1])) == 1
        assert np.unique(img[:, :, 1])[0] == 0

        # Red channel should have at most two values: 0 and 128 for boundaries
        assert len(np.unique(img[:, :, 2])) <= 2
        assert np.unique(img[:, :, 2])[0] == 0
        if len(np.unique(img[:, :, 2])) > 1:
            assert np.unique(img[:, :, 2])[1] == 128

        ##################################################################
        # Prepare the image

        # Find and remove boundaries: this is necessary as they are marked with 8 in the blue channel as well
        locations = np.where(img == 128)
        img[locations[0], locations[1]] = 0
        # Find regular text and text + decoration
        locations_text = np.where(img == 8)
        locations_text_decoration = np.where(img == 12)
        # Wipe the image
        img[:, :, :] = 0
        # Set the text to be white
        img[locations_text[0], locations_text[1]] = 255
        img[locations_text_decoration[0], locations_text_decoration[1]] = 255

    # Rotate 90 degrees to the left the image (for vertical scripts such as Chinese)
    if vertical:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return img
