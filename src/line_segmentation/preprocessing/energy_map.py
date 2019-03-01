import logging
import sys
import time

import cv2
import numpy as np
from scipy.spatial import distance
from skimage import measure

from src.line_segmentation.utils.unused_but_keep_them import blur_image
from src.line_segmentation.utils.util import calculate_asymmetric_distance, save_img


def create_heat_map_visualization(ori_energy_map):
    # -------------------------------
    start = time.time()
    # -------------------------------

    heatmap = ((np.copy(ori_energy_map) / np.max(ori_energy_map)))
    heatmap = (np.stack((heatmap,) * 3, axis=-1)) * 255
    heatmap = np.array(heatmap, dtype=np.uint8)
    # show_img(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # result = cv2.add(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), img)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return heatmap


def prepare_energy(ori_map, left_column, right_column, y):
    """
    Sets the left and right border of the matrix to int.MAX except at y.

    :param ori_map:
    :param y:
    :return:
    """

    y_value_left, y_value_right = left_column[y], right_column[y]
    ori_map[:, 0] = sys.maxsize / 2
    ori_map[:, -1] = sys.maxsize / 2

    ori_map[y][0], ori_map[y][-1] = y_value_left, y_value_right

    return ori_map

def create_distance_matrix(img_shape, centroids, asymmetric=False, side_length=1000):
    # -------------------------------
    start = time.time()
    # -------------------------------

    template = np.zeros((side_length, side_length))
    center_template = np.array([[np.ceil(side_length / 2), np.ceil(side_length / 2)]])
    pixel_coordinates = np.asarray([[x, y] for x in range(template.shape[0]) for y in range(template.shape[1])])

    # calculate distance template
    # TODO save template for speed up
    if asymmetric:
        template = np.array([calculate_asymmetric_distance(center_template, pxl, 1, 5) for pxl in pixel_coordinates]) \
            .flatten().reshape((side_length, side_length))
    else:
        template = distance.cdist(center_template, pixel_coordinates).flatten().reshape((side_length, side_length))

    # show_img(create_heat_map_visualization(template))

    distance_matrix = np.ones(img_shape) * np.max(template)
    # show_img(create_heat_map_visualization(template))
    # template[template > np.ceil(side_length / 2)] = np.max(template) * 2
    # round the centroid coordinates to ints to use them as array index
    centroids = np.rint(centroids).astype(int)

    # for each centroid
    for centroid in centroids:
        pos_v, pos_h = (centroid - np.ceil(side_length / 2)).astype(int)  # offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + template.shape[0], distance_matrix.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + template.shape[1], distance_matrix.shape[1]), 0))

        v_range2 = slice(max(0, -pos_v), min(-pos_v + distance_matrix.shape[0], template.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + distance_matrix.shape[1], template.shape[1]))

        # need max
        distance_matrix[v_range1, h_range1] = np.minimum(template[v_range2, h_range2],
                                                         distance_matrix[v_range1, h_range1])

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    # distance_matrix = distance_matrix.reshape((centroids.shape[0], img_shape[0] * img_shape[1]))
    return distance_matrix.flatten()


def create_energy_map(img, blurring=True, projection=True, asymmetric=False):
    # -------------------------------
    start = time.time()
    # -------------------------------
    # get the cc, all the centroids and the areas of the cc
    cc, centroids, areas = find_cc_centroids_areas(img)

    # a list of all the pixels in the image as tuple
    centroids = np.asarray([[point[0], point[1]] for point in centroids])

    # normalise between 0 nd 1
    areas = (areas - np.min(areas)) / (np.max(areas) - np.min(areas))

    # bring it between -1 and 1
    areas = areas - np.mean(areas)

    # make it all negative
    areas = - np.abs(areas)

    # scale it with punishment
    areas *= 500

    # creating distance matrix
    # pixel_coordinates = np.asarray([[x, y] for x in range(img.shape[0]) for y in range(img.shape[1])])
    # distance_matrix = distance.cdist(pixel_coordinates, centroids[0:10])
    distance_matrix = create_distance_matrix(img.shape[0:2], centroids, asymmetric=asymmetric)

    # cap the distance to >= 1
    # distance_matrix[np.where(distance_matrix < 1)] = 1

    # scale down the distance
    distance_matrix /= 30

    # make sure the distance is > 1
    distance_matrix += 1

    # We give all centroids the same energy (100)
    energy_background = ((np.ones(img.shape[0] * img.shape[1]) * 100) / distance_matrix).transpose()
    # energy_background = ((np.ones(areas.shape) * 100) / distance_matrix).transpose()
    # energy_background = np.max(energy_background, axis=0)
    # get the text location
    locs = np.array(np.where(img[:, :, 0].reshape(-1) == 0))[0:2, :]
    energy_text = energy_background / 2
    energy_text[locs] = 0

    # optional to speed up the method (get all pixels which are in a certain distance of a centroid)
    # get distance between each pixel and each centroid (because gravity)
    # sum up the received energy for each pixel
    energy_map = energy_background + energy_text
    energy_map = energy_map.reshape(img.shape[0:2])

    if blurring:
        # blur the map
        blurred_energy_map = blur_image(img=energy_map, filter_size=300)
        energy_map = blurred_energy_map + energy_text.reshape(img.shape[0:2])

    if projection:
        projection_profile = create_projection_profile(energy_map)
        # scale it between 0 and max(energy_map) / 2
        projection_profile *= np.max(energy_map) / 2

        # blur projection profile
        projection_matrix = np.zeros(img.shape[0:2])
        projection_matrix = (projection_matrix.transpose() + projection_profile).transpose()
        projection_matrix = blur_image(projection_matrix, filter_size=1000)

        # overlap it with the normal energy map and add the text energy
        energy_map = energy_map + projection_matrix + energy_text.reshape(img.shape[0:2])

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return energy_map, cc


def create_projection_profile(energy_map):
    # creating the horizontal projection profile
    pp = np.sum(energy_map, axis=1)
    # smoothing it
    WINDOW_SIZE = 100
    pp = smooth(pp, WINDOW_SIZE)[int(WINDOW_SIZE/2):-int(WINDOW_SIZE/2-1)]
    # wipe everything below average
    pp -= np.mean(pp)
    pp[pp < 0] = 0
    # normalize it between 0-1
    pp = (pp - np.min(pp)) / (np.max(pp) - np.min(pp))
    return pp


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y



def find_cc_centroids_areas(img):
    # -------------------------------
    start = time.time()
    # -------------------------------
    #############################################
    # Find CC
    cc_labels, cc_properties = get_connected_components(img)

    amount_of_properties = 0

    # Compute average metrics
    avg_area = np.mean([item.area for item in cc_properties])
    std_area = np.std([item.area for item in cc_properties])
    avg_height = np.mean([item.bbox[2] - item.bbox[0] for item in cc_properties])
    avg_width = np.mean([item.bbox[3] - item.bbox[1] for item in cc_properties])

    while amount_of_properties != len(cc_properties):
        # for _ in range(2):
        amount_of_properties = len(cc_properties)
        image = img[:, :, 1]
        #############################################
        # Cut all large components into smaller components
        coef = 1.5
        for item in cc_properties:
            if item.area > coef * avg_area \
                    or item.bbox[2] - item.bbox[0] > coef * avg_height \
                    or item.bbox[3] - item.bbox[1] > coef * avg_width:
                v_size = abs(item.bbox[0] - item.bbox[2])
                h_size = abs(item.bbox[1] - item.bbox[3])
                y1, x1, y2, x2 = item.bbox

                if float(h_size) / v_size > 1.5:
                    image[y1:y2, np.round((x1 + x2) / 2).astype(int)] = 0
                elif float(v_size) / h_size > 1.5:
                    image[np.round((y1 + y2) / 2).astype(int), x1:x2] = 0
                else:
                    # img[np.round((y1 + y2) / 2).astype(int), np.round((x1 + x2) / 2).astype(int)] = 0
                    image[y1:y2, np.round((x1 + x2) / 2).astype(int)] = 0
                    image[np.round((y1 + y2) / 2).astype(int), x1:x2] = 0

        img[:, :, 1] = image

        # Re-find CC
        cc_labels, cc_properties = get_connected_components(img)
        #############################################

    # Collect CC centroids
    all_centroids = np.asarray([cc.centroid[0:2] for cc in cc_properties])

    # Collect CC sizes
    all_areas = np.asarray([cc.area for cc in cc_properties])

    # Discard outliers & sort
    no_outliers = detect_outliers(all_areas, avg_area, std_area)
    centroids = all_centroids[no_outliers, :]
    filtered_area = all_areas[no_outliers]
    all_areas = filtered_area[np.argsort(centroids[:, 0])]
    all_centroids = centroids[np.argsort(centroids[:, 0]), :]

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return (cc_labels, cc_properties), all_centroids, all_areas


def get_connected_components(img):
    cc_labels = measure.label(img[:, :, 1], background=0)
    cc_properties = measure.regionprops(cc_labels, cache=True)
    return cc_labels, cc_properties


def detect_outliers(area, mean, std):
    # -------------------------------
    start = time.time()
    # -------------------------------
    if mean is not None:
        mean = np.mean(area)
    if std is not None:
        std = np.std(area)

    #no_outliers = abs(area - mean) < 3 * std
    no_outliers = area - 0.25*mean > 0

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return no_outliers