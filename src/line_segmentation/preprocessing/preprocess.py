import sys

import cv2
import logging
import time
import matplotlib
from scipy import signal
from skimage import measure

import src
from src.line_segmentation.line_segmentation import get_connected_components
from src.line_segmentation.preprocessing.energy_map import find_cc_centroids_areas
from src.line_segmentation.utils.util import save_img

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def preprocess(image):
    # -------------------------------
    start = time.time()
    # -------------------------------

    # Save a copy of the image s.t we can return it at the end
    original = image

    # Get only green channel
    image = image[:, :, 1]

    # find the text area and wipe the rest
    image = wipe_outside_textarea(image)

    # Remove components which are too small in terms of area
    #image = remove_small_components(image)


    # Put together the new green channel into the original and return it. We must return an RBG image.
    original[:, :, 1] = image

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------
    return original


def intersection(lst1, lst2):
    return [value for value in lst1 if value in  set(lst2)]


def wipe_outside_textarea(image):
    # SMOOTH IMAGE ######################################################################
    filter_size = 128
    kernel = np.ones((filter_size, filter_size)) / filter_size
    # Apply averaging filter
    image = cv2.filter2D(image, -1, kernel)

    # GET BIGGEST COMPONENT #############################################################
    # Find CC
    cc_labels, cc_properties = get_connected_components(image)



    # Get contour points of the binary polygon image
    cc = measure.find_contours(image, 254, fully_connected='high')[0]


    # take the biggest polygon
  #  contour = polygon_coords[nb_line][0]

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

    return image


def remove_small_components(image):

    pass
