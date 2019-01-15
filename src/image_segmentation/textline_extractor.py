"""
"""

# Utils
import logging
import os
import time

import cv2
import numpy as np
import sys
from XMLhandler import writePAGEfile
from scipy.spatial import ConvexHull, distance
from skimage import measure, transform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

#######################################################################################################################
from src.image_segmentation.seamcarving import horizontal_seam, draw_seam, build_seam_energy_map


def main(input_loc, show_seams=False, show_heatmap=False, penalty=3000):
    """
    Function to compute the text lines from a segmented image. This is the main routine where the magic happens
    :param input_loc: path to segmented image
    :param output_loc: path to save generated PAGE XML
    :param a: param_a
    :param b: param_b
    """

    # print("{}".format(read_max_textline_from_file('./../data/e-codices_fmb-cb-0055_0019r_max_gt.xml')))

    # -------------------------------
    start_whole = time.time()
    # -------------------------------

    #############################################
    # Load the image
    img = cv2.imread(input_loc)

    # blow up image with the help of seams
    img = textline_separation(img, penalty, show_heatmap, show_seams, start_whole, nb_of_iterations=3)

    # validate
    measure_separation(img)

#######################################################################################################################


def textline_separation(img, penalty, show_heatmap, show_seams, start_whole, nb_of_iterations=5):
    """
    Contains the main loop. In each iteration it creates an energy map based on the given image CC and
    blows it up.

    :param img:
    :param penalty:
    :param show_heatmap:
    :param show_seams:
    :param start_whole:
    :return:
    """

    for i in range(nb_of_iterations):
        if i == 0:
            # Prepare image (filter only text, ...)
            img = prepare_image(img, cropping=False)

        # create the engergy map
        ori_energy_map = create_energy_map(img, blurring=False, projection=True, asymmetric=False)

        # bidirectional energy map
        # bi_energy_map = build_seam_energy_map(ori_energy_map)

        # show_img((ori_enegery_map/max_en) * 255)
        # energy_map_representation = np.copy(ori_energy_map)

        if show_heatmap:
            # visualize the energy map as heatmap
            heatmap = create_heat_map_visualization(ori_energy_map)
        else:
            heatmap = np.copy(ori_energy_map)

        # list with all seams
        seams = []

        # show_img(ori_enegery_map)
        for seam_at in range(0, img.shape[0], 5):
            energy_map = prepare_energy(ori_energy_map, seam_at)

            seam = horizontal_seam(energy_map, penalty=True, penalty_div=penalty)
            seams.append(seam)
            if show_seams:
                draw_seam(heatmap, seam)

        # -------------------------------
        stop_whole = time.time()
        logging.info("finished after: {diff} s".format(diff=stop_whole - start_whole))
        # -------------------------------

        img, growth = blow_up_image(img, seams)

        penalty += penalty * growth

        show_img(heatmap, save=True, name='../results/blow_up_without_seams/blow_up_{i}.png'.format(i=i),
                 show=show_heatmap)

    return img


def measure_separation(img):
    # create projection profile

    pass



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
    logging.info("finished after: {diff} s".format(diff=stop-start))
    # -------------------------------

    return heatmap


def prepare_image(img, cropping=True):
    # -------------------------------
    start = time.time()
    # -------------------------------

    img[:, :, 0] = 0
    img[:, :, 2] = 0
    locations = np.where(img == 127)
    img[:, :, 1] = 0
    img[locations[0], locations[1]] = 255
    if cropping:
        locs = np.array(np.where(img == 255))[0:2, ]
        img = img[np.min(locs[0, :]):np.max(locs[0, :]), np.min(locs[1, :]):np.max(locs[1, :])]

    # # Erase green (if any, but shouldn't have values here)
    # img[:, :, 1] = 0
    # # Find and remove boundaries (set to bg)
    # boundaries = np.where(img == 128)
    # img[boundaries[0], boundaries[1]] = 0
    # # Find regular text and make it white
    # locations = np.where(img == 8)
    # img[locations[0], locations[1]] = 128
    # # Find text+decoration and make it white
    # locations = np.where(img == 12)
    # img[locations[0], locations[1]] = 128
    # # Erase red & blue (so we get rid of everything else, only green will be set)
    # img[:, :, 0] = 0
    # img[:, :, 2] = 0

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop-start))
    # -------------------------------

    return img


def blow_up_image(image, seams):
    # new image
    new_image = []

    # get the new height of the image and the original one
    ori_height, _, _ = image.shape
    height = ori_height + len(seams)

    seams = np.array(seams)

    for i in range(0, image.shape[1]):
        ori_col = image[:, i]
        col = np.copy(image[:, i])
        y_cords_seams = seams[:, i, 1]

        seam_nb = 0
        for y_seam in y_cords_seams:
            col = np.insert(col, y_seam + seam_nb, [0, 0, 0], axis=0)
            seam_nb += 1

        new_image.append(col)

    return np.swapaxes(np.asarray(new_image), 0, 1), (((100 / ori_height) * height) - 100) / 100


def cut_img(img, cc_props):
    # -------------------------------
    start = time.time()
    # -------------------------------

    avg_area = np.mean([item.area for item in cc_props])
    avg_height = np.mean([item.bbox[2] - item.bbox[0] for item in cc_props])
    avg_width = np.mean([item.bbox[3] - item.bbox[1] for item in cc_props])
    for item in cc_props:
        if item.area > 2.8 * avg_area or item.bbox[2] - item.bbox[0] > 2.8 * avg_height or item.bbox[3] - item.bbox[
            1] > 2.8 * avg_width:
            v_size = abs(item.bbox[0] - item.bbox[2])
            h_size = abs(item.bbox[1] - item.bbox[3])
            y1, x1, y2, x2 = item.bbox

            if float(h_size) / v_size > 1.5:
                img[y1:y2, np.round((x1 + x2) / 2).astype(int)] = 0
            elif float(v_size) / h_size > 1.5:
                img[np.round((y1 + y2) / 2).astype(int), x1:x2] = 0
            else:
                # img[np.round((y1 + y2) / 2).astype(int), np.round((x1 + x2) / 2).astype(int)] = 0
                img[y1:y2, np.round((x1 + x2) / 2).astype(int)] = 0
                img[np.round((y1 + y2) / 2).astype(int), x1:x2] = 0

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return img


def detect_outliers(area):
    # -------------------------------
    start = time.time()
    # -------------------------------

    too_big = abs(area + np.mean(area)) < 5 * np.std(area)
    too_small = abs(area - np.mean(area)) < 5 * np.std(area)

    # too_big = area > (0.4 * np.mean(area))
    # too_small = area < 3 * np.mean(area)
    # too_small = area > 0
    # no_y = abs(centroids - np.mean(centroids)) < 2 * np.std(centroids)

    no_outliers = [x & y for (x, y) in zip(too_big, too_small)]

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return no_outliers


def cluster(img, centroids, area, eps, min_samples):
    # import matplotlib.pyplot as plt

    # Attempt clustering with DBSCAN
    X = centroids[:, 0]
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    area = (area - np.min(area)) / (np.max(area) - np.min(area))
    tmp = np.zeros((X.shape[0], 2))
    # tmp[:, 1] = area
    tmp[:, 0] = X

    # plt.figure()
    # plt.scatter(X, area)
    # plt.show()

    X = tmp

    # eps = 0.01  # centroids test1&2&3&4 (min sample 5) GT=4,16,13,29

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    # Draw DBSCAN outliers
    for i in range(0, len(centroids)):
        # Draw outliers
        if db.labels_[i] == -1:
            cv2.circle(img, tuple([75, np.round(centroids[i, 0]).astype(np.int)]), radius=3,
                       color=(0, 0, 127), thickness=3, lineType=1, shift=0)
            # print('{}'.format(i))

    # Remove outliers
    centroids = centroids[db.labels_ != -1, :]
    labels = db.labels_[db.labels_ != -1]
    return centroids, labels


def draw_clusters(img, centroids, labels):
    # Draw and count the cluster
    n_clusters = 1
    clusters_lines = []
    for i in range(0, len(centroids) - 1):
        # Draw the line between the clusters
        if labels[i] != labels[i + 1]:
            # Count the cluster
            n_clusters += 1
            # Compute line location
            clusters_lines.append((centroids[i, 0] + centroids[i + 1, 0]) / 2.0)
            # Draw line
            cv2.line(img, tuple([0, np.round(clusters_lines[n_clusters - 2]).astype(np.int)]),
                     tuple([4000, np.round(clusters_lines[n_clusters - 2]).astype(np.int)]),
                     color=(0, 127, 0), thickness=4, lineType=8, shift=0)
    return clusters_lines


def find_cc_from_centroid(c, cc_properties):
    for cc in cc_properties:
        if (np.asarray(cc.centroid[0:2]) == c).all():
            return cc
    return None


def points_in_line(cc_properties, line, fast=False):
    points = []
    if fast:
        points.extend(find_cc_from_centroid(line[0], cc_properties).coords[::3, 0:2])
        points.extend(find_cc_from_centroid(line[-1], cc_properties).coords[::3, 0:2])
    else:
        for c in line:
            cc = find_cc_from_centroid(c, cc_properties)
            points.extend(cc.coords[::3, 0:2])
    points = np.array(points)
    return points


def separate_in_bins(centroids, clusters_lines):
    clusters_centroids = [[]]
    l = 0
    for c in zip(centroids):
        if l == len(clusters_lines) or c[0][0] < clusters_lines[l]:
            clusters_centroids[l].append(c[0])
        else:
            l += 1
            clusters_centroids.append([])
            clusters_centroids[l].append(c[0])

    # Sort the bins according to the horizontal axis
    for i in range(0, len(clusters_centroids)):
        clusters_centroids[i] = np.asarray(clusters_centroids[i])
        clusters_centroids[i] = clusters_centroids[i][np.argsort(clusters_centroids[i][:, 1]), :]

    return clusters_centroids


def blur_image(img, save_name="blur_image.png", save=False, show=False, filter_size=1000, horizontal=True):
    # motion blur the image
    # generating the kernel
    kernel_motion_blur = np.zeros((filter_size, filter_size))
    if horizontal:
        kernel_motion_blur[int((filter_size - 1) / 2), :] = np.ones(filter_size)
    else:
        kernel_motion_blur[:, int((filter_size - 1) / 2)] = np.ones(filter_size)
    kernel_motion_blur = kernel_motion_blur / filter_size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)

    if save:
        cv2.imwrite(save_name, output)

    if show:
        cv2.imshow('image', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output  #, np.sum(output, axis=2)


def calculate_asymmetric_distance(x, y, h_weight=1, v_weight=5):
    return [np.sqrt(((y[0] - x[0][0]) ** 2) * v_weight + ((y[1] - x[0][1]) ** 2) * h_weight)]


def create_distance_matrix(img_shape, centroids, asymmetric=False, side_length=1000):
    # -------------------------------
    start = time.time()
    # -------------------------------

    template = np.zeros((side_length, side_length))
    center_template = np.array([[np.ceil(side_length / 2), np.ceil(side_length / 2)]])
    pixel_coordinates = np.asarray([[x, y] for x in range(template.shape[0]) for y in range(template.shape[1])])

    # calculate template
    # TODO save template for speed up
    if asymmetric:
        template = np.array([calculate_asymmetric_distance(center_template, pxl) for pxl in pixel_coordinates])\
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
        distance_matrix[v_range1, h_range1] = np.minimum(template[v_range2, h_range2], distance_matrix[v_range1, h_range1])
        # show_img(create_heat_map_visualization(distance_matrix))

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    # that the rest has not a high energy
    distance_matrix[np.where(distance_matrix == 1)] = np.max(template)

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
    distance_matrix /= 15

    # make sure the distance is > 0
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

    return energy_map


def create_projection_profile(map):
    # creating the horizontal projection profile
    projection_profile = np.sum(map, axis=1)
    # normalize it between 0-1
    projection_profile = (projection_profile - np.min(projection_profile)) / (
                np.max(projection_profile) - np.min(projection_profile))
    # scale it between 0 and max(energy_map) / 2
    projection_profile *= np.max(map) / 2
    return projection_profile


def find_cc_centroids_areas(img):
    # -------------------------------
    start = time.time()
    # -------------------------------
    #############################################
    # Find CC
    cc_labels = measure.label(img[:, :, 1], background=0)
    cc_properties = measure.regionprops(cc_labels, cache=True)

    amount_of_properties = 0

    while amount_of_properties != len(cc_properties):
        # for _ in range(2):
        amount_of_properties = len(cc_properties)
        #############################################
        # Cut all large components into smaller components
        img[:, :, 1] = cut_img(img[:, :, 1], cc_properties)

        # Re-find CC
        cc_labels = measure.label(img[:, :, 1], background=0)
        cc_properties = measure.regionprops(cc_labels, cache=True)
        #############################################

    # Collect CC centroids
    all_centroids = np.asarray([cc.centroid[0:2] for cc in cc_properties])
    all_centroids = all_centroids[np.argsort(all_centroids[:, 0]), :]

    # Collect CC sizes
    all_areas = np.asarray([cc.area for cc in cc_properties])

    # Discard outliers & sort
    no_outliers = detect_outliers(all_areas)
    centroids = all_centroids[no_outliers, :]
    filtered_area = all_areas[no_outliers]
    all_areas = filtered_area[np.argsort(centroids[:, 0])]
    all_centroids = centroids[np.argsort(centroids[:, 0]), :]

    # discard outliers
    # big_enough = all_areas > 0.4 * np.mean(all_areas)
    # small_enough = all_areas > 0
    # no_outliers = [x & y for (x, y) in zip(big_enough, small_enough)]
    # all_centroids = all_centroids[no_outliers, :]
    #
    # all_areas = all_areas[no_outliers]

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return (cc_labels, cc_properties), all_centroids, all_areas


def prepare_energy(ori_map, y):
    """
    Sets the left and right border of the matrix to int.MAX except at y.

    :param ori_map:
    :param y:
    :return:
    """
    energy_map = np.copy(ori_map)

    for row in range(energy_map.shape[0]):
        if row == y:
            continue

        energy_map[row][0] = sys.maxsize / 2
        energy_map[row][energy_map.shape[1] - 1] = sys.maxsize / 2

    return energy_map


def show_img(img, save=False, name='experiment.png', show=True):
    if show:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(name, img)


#######################################################################################################################

if __name__ == "__main__":
    # Set up logging to console
    formatter = logging.Formatter(
        fmt='%(asctime)s %(funcName)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    stderr_handler = logging.StreamHandler()
    stderr_handler.formatter = formatter
    logging.getLogger().addHandler(stderr_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
        level=logging.INFO)
    logging.info('Printing activity to the console')

    print("{}".format(os.getcwd()))
    main(input_loc='../data/test1.png')
    logging.info('Terminated')
