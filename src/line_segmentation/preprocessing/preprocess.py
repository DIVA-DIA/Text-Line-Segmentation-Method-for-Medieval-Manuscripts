import logging
import os
import sys
import time

import cv2
from skimage import measure

import matplotlib

from src.line_segmentation.utils.util import save_img

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def preprocess(image):
    # -------------------------------
    start = time.time()
    # -------------------------------

    # find the text area and wipe the rest
    image = wipe_outside_textarea(image)

    # Remove components which are too small in terms of area
    image = remove_small_components(image)

    # Remove components which are too big in terms of area -> after removing the small ones!
    image = remove_big_components(image)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------
    return image


def wipe_outside_textarea(image):

    # Save a copy of the original image
    ORIGINAL = image

    # Get only green channel
    image = image[:, :, 1]

    # SMOOTH IMAGE ######################################################################
    filter_size_H = 64
    filter_size_V = 192
    kernel = np.ones((filter_size_V, filter_size_H)) / filter_size_H
    # Apply averaging filter
    image = cv2.filter2D(image, -1, kernel)
    #SMOOTH_IMAGE = image
    # Draw a vertical line in the middle of the image to prevent 2 paragraphs to be split
    image[5:-5, int(image.shape[1] / 2) - 5:int(image.shape[1] / 2) + 5] = 255

    # GET BIGGEST COMPONENT #############################################################
    # Get contour points of the binary polygon image
    tmp = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cc = measure.find_contours(image, 200, fully_connected='high')[0]
    # Swap the columns of cc as the coordinate are reversed
    cc[:, 0], cc[:, 1] = cc[:, 1], cc[:, 0].copy()
    # Cast to int to make, once again, cv2.fillPoly() happy
    cc = [cc.astype(np.int32, copy=False)]
    cv2.fillPoly(tmp, cc, (255, 255, 255))
    # DEBUG
    #save_img(tmp, path=os.path.join('./output', 'smoothed_image.png'), show=False)

    # WIPE EVERYTHING OUTSIDE THIS AREA #################################################
    # Use 'tmp' as mask on the original image. Pixel with value '0' are text.
    tmp = tmp - ORIGINAL
    # Prepare image in RBG format s.t. we can use the coordinates systems of tmp
    image = np.stack((image,) * 3, axis=-1)
    # Wipe the pixels which are not selected by the mask
    image[np.where(tmp != 0)] = 0
    # DEBUG
    #save_img(image, path=os.path.join('./output', 'filtered_image.png'), show=False)

    """
    # FILTER WITH VERTICAL PROJECTION PROFILE ###########################################
    # Compute projection profile
    ver = np.sum(SMOOTH_IMAGE, axis=0)
    # Get all values above average
    ver_indexes = np.where(ver > np.mean(ver))
    # Find the first and last of them
    left = np.min(ver_indexes)
    right = np.max(ver_indexes)

    # Wipe the image on left/right sides
    image[:, 0:left] = 0
    image[:, right:] = 0

    plt.figure()
    plt.plot(ver)
    plt.axhline(y=np.mean(ver), color='r', linestyle='-')
    plt.axvline(x=left, color='r', linestyle='-')
    plt.axvline(x=right, color='r', linestyle='-')
    plt.savefig('./output/ver.png')

    # FILTER WITH VERTICAL PROJECTION PROFILE ###########################################
    # Compute projection profile
    hor = np.sum(SMOOTH_IMAGE, axis=1)
    # Get all values above average
    hor_indexes = np.where(hor > np.mean(hor))
    # Find the first and last of them
    top = np.min(hor_indexes)
    bottom = np.max(hor_indexes)

    # Wipe the image on top/bottom sides
    image[0:top, :] = 0
    image[bottom:, :] = 0

    plt.figure()
    plt.plot(hor)
    plt.axhline(y=np.mean(hor), color='r', linestyle='-')
    plt.axvline(x=top, color='r', linestyle='-')
    plt.axvline(x=bottom, color='r', linestyle='-')
    plt.savefig('./output/hor.png')
    """
    return image


def remove_small_components(image):
    # Find CC
    cc_properties = measure.regionprops(measure.label(image[:, :, 1], background=0), cache=True)

    # Compute average metrics
    avg_area = np.mean([item.area for item in cc_properties])

    # Remove all small components
    for cc in cc_properties:
        if cc.area < 0.1 * avg_area:
            # Wipe the cc
            image[(cc.coords[:, 0], cc.coords[:, 1])] = 0
    return image


def remove_big_components(image):
    # Find CC
    cc_properties = measure.regionprops(measure.label(image[:, :, 1], background=0), cache=True)

    # Compute average metrics
    avg_area = np.mean([item.area for item in cc_properties])

    # Remove all small components
    for cc in cc_properties:
        if cc.area > 10 * avg_area:
            # Wipe the cc
            image[(cc.coords[:, 0], cc.coords[:, 1])] = 0
    return image
