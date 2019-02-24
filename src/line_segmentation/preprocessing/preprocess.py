import logging
import os
import time

import cv2
import matplotlib
from skimage import measure

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
    #image = remove_small_components(image)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------
    return image


def intersection(lst1, lst2):
    return [value for value in lst1 if value in  set(lst2)]


def wipe_outside_textarea(image):

    # Save a copy of the original image
    ORIGINAL = image

    # Get only green channel
    image = image[:, :, 1]

    # SMOOTH IMAGE ######################################################################
    filter_size = 128
    kernel = np.ones((filter_size, filter_size)) / filter_size
    # Apply averaging filter
    image = cv2.filter2D(image, -1, kernel)
    # Draw a vertical line in the middle of the image to prevent 2 paragraphs to be split
    save_img(np.expand_dims(image, 2), path=os.path.join('./output', 'test_b.png'), show=False)
    image[5:-5, int(image.shape[1] / 2) - 5:int(image.shape[1] / 2) + 5] = 255
    save_img(np.expand_dims(image, 2), path=os.path.join('./output', 'test_a.png'), show=False)

    # GET BIGGEST COMPONENT #############################################################
    # Get contour points of the binary polygon image
    tmp = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cc = measure.find_contours(image, 200, fully_connected='high')[0]
    # Swap the columns of cc as the coordinate for cv2.fillPoly() are reversed
    cc[:, 0], cc[:, 1] = cc[:, 1], cc[:, 0].copy()
    # Cast to int to make, once again, cv2.fillPoly() happy
    cc = [cc.astype(np.int32, copy=False)]
    cv2.fillPoly(tmp, cc, (255, 255, 255))
    # DEBUG
    save_img(tmp, path=os.path.join('./output', 'smoothed_image.png'), show=False)

    # WIPE EVERYTHING OUTSIDE THIS AREA ################################################
    # Use 'tmp' as mask on the original image. Pixel with value '0' are text.
    tmp = tmp - ORIGINAL
    # Prepare image in RBG format s.t. we can use the coordinates systems of tmp
    image = np.stack((image,) * 3, axis=-1)
    # Wipe the pixels which are not selected by the mask
    image[np.where(tmp != 0)] = 0
    # DEBUG
    save_img(image, path=os.path.join('./output', 'filtered_image.png'), show=False)

    """
    # FILTER WITH VERTICAL PROJECTION PROFILE ###########################################
    # Compute projection profile
    ver = np.sum(image, axis=0)
    # Get all values above average
    ver_indexes = np.where(ver > np.mean(ver))
    # Find the first and last of them
    left = np.min(ver_indexes)
    right = np.max(ver_indexes)

    # Wipe the image on left/right sides
    # image[:, 0:left] = 0
    # image[:, right:] = 0

    plt.figure()
    plt.plot(ver)
    plt.axhline(y=np.mean(ver), color='r', linestyle='-')
    plt.axvline(x=left, color='r', linestyle='-')
    plt.axvline(x=right, color='r', linestyle='-')
    plt.savefig('./output/ver.png')

    # FILTER WITH VERTICAL PROJECTION PROFILE ###########################################
    # Compute projection profile
    hor = np.sum(image, axis=1)
    # Get all values above average
    hor_indexes = np.where(hor > np.mean(hor))
    # Find the first and last of them
    top = np.min(hor_indexes)
    bottom = np.max(hor_indexes)

    # Wipe the image on top/bottom sides
    # image[0:top, :] = 0
    # image[bottom:, :] = 0

    plt.figure()
    plt.plot(hor)
    plt.axhline(y=np.mean(hor), color='r', linestyle='-')
    plt.axvline(x=top, color='r', linestyle='-')
    plt.axvline(x=bottom, color='r', linestyle='-')
    plt.savefig('./output/hor.png')
    """
    return image


def remove_small_components(image):
    pass
