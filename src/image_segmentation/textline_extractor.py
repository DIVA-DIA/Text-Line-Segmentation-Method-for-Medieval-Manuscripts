"""
"""

# Utils
import logging
import os

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


def segment_textlines(input_loc):
    """
    Function to compute the text lines from a segmented image. This is the main routine where the magic happens
    :param input_loc: path to segmented image
    :param output_loc: path to save generated PAGE XML
    :param a: param_a
    :param b: param_b
    """

    # print("{}".format(read_max_textline_from_file('./../data/e-codices_fmb-cb-0055_0019r_max_gt.xml')))

    #############################################
    # Load the image
    img = cv2.imread(input_loc)

    # Prepare image (filter only text, ...)
    img = prepare_image(img, cropping=False)

    # TODO for testing
    # blur_energy_map_sc(img)

    # create the engergy map
    ori_energy_map = create_energy_map(img, blurring=False, projection=True)

    # bidirectional energy map
    # bi_energy_map = build_seam_energy_map(ori_energy_map)

    # show_img((ori_enegery_map/max_en) * 255)
    energy_map_representation = np.copy(ori_energy_map)

    # visualize the energy map as heatmap
    heatmap = create_heat_map_visualization(ori_energy_map)
    # heatmap = create_heat_map_visualization(creating_ellipsoid(img))

    # show_img(ori_enegery_map)
    for i in range(0, img.shape[0], 20):
        energy_map = prepare_energy(ori_energy_map, i)

        # non-library-seam carving
        test = horizontal_seam(energy_map)
        draw_seam(heatmap, test)

    show_img(heatmap, save=False)


#######################################################################################################################


def create_heat_map_visualization(ori_energy_map):
    heatmap = ((np.copy(ori_energy_map) / np.max(ori_energy_map)))
    heatmap = (np.stack((heatmap,) * 3, axis=-1)) * 255
    heatmap = np.array(heatmap, dtype=np.uint8)
    # show_img(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # result = cv2.add(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), img)
    return heatmap


def prepare_image(img, cropping=True):
    img[:, :, 0] = 0
    img[:, :, 2] = 0
    locations = np.where(img == 127)
    img[:, :, 1] = 0
    img[locations[0], locations[1]] = 255
    if cropping:
        locs = np.array(np.where(img == 255))[0:2, ]
        img = img[np.min(locs[0, :]):np.max(locs[0, :]), np.min(locs[1, :]):np.max(locs[1, :])]

    # Erase green (if any, but shouldn't have values here)
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
    return img


def cut_img(img, cc_props):
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

    return img


def detect_outliers(area):
    too_big = abs(area + np.mean(area)) < 5 * np.std(area)
    too_small = abs(area - np.mean(area)) < 5 * np.std(area)

    # too_big = area > (0.4 * np.mean(area))
    # too_small = area < 3 * np.mean(area)
    # too_small = area > 0
    # no_y = abs(centroids - np.mean(centroids)) < 2 * np.std(centroids)

    no_outliers = [x & y for (x, y) in zip(too_big, too_small)]
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


def create_energy_map(img, blurring=True, projection=True):
    # get the cc, all the centroids and the areas of the cc
    cc, centroids, areas = find_cc_centroids_areas(img)

    # a list of all the pixels in the image as tuple
    pixel_coordinates = np.asarray([[x, y] for x in range(img.shape[0]) for y in range(img.shape[1])])
    centroids = np.asarray([[point[0], point[1]] for point in centroids])

    # normalise between 0 nd 1
    areas = (areas - np.min(areas)) / (np.max(areas) - np.min(areas))

    # bring it between -1 and 1
    areas = areas - np.mean(areas)

    # make it all negative
    areas = - np.abs(areas)

    # scale it with punishment
    areas *= 500

    # build ellipsoid
    # creating_ellipsoid(img)

    # multiply all the distances of a certain centroid with the area as weight.
    # metric = lambda u, v: np.sqrt(np.sum((h_weight * u - v_weight * v)**2))
    # metric = np.vectorize(metric)
    # distance_matrix = [metric(centroid, pixel) for centroid in centroids for pixel in pixel_coordinates]
    distance_matrix = distance.cdist(pixel_coordinates, centroids)

    # cap the distance to >= 1
    distance_matrix[np.where(distance_matrix < 1)] = 1

    # scale down the distance
    distance_matrix /= 15

    # make sure the distance is > 0
    distance_matrix += 1

    # We give all centroids the same energy (100)
    energy_background = ((np.ones(areas.shape) * 100) / distance_matrix).transpose()
    energy_background = np.max(energy_background, axis=0)
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
        # creating the horizontal projection profile
        projection_profile = np.sum(energy_map, axis=1)
        # normalize it between 0-1
        projection_profile = (projection_profile - np.min(projection_profile)) / (np.max(projection_profile) - np.min(projection_profile))
        # scale it between 0 and max(energy_map) / 2
        projection_profile *= np.max(energy_map) / 2

        # blur projection profile
        projection_matrix = np.zeros(img.shape[0:2])
        projection_matrix = (projection_matrix.transpose() + projection_profile).transpose()
        projection_matrix = blur_image(projection_matrix, filter_size=1000)

        # overlap it with the normal energy map and add the text energy
        energy_map = energy_map + projection_matrix + energy_text.reshape(img.shape[0:2])

    return energy_map


def creating_ellipsoid(img):
    # horizontal and vertical weights
    h_weight = 1
    v_weight = 50

    # a list with all pixel coordinates
    pixel_coordinates = np.asarray([[x, y] for x in range(img.shape[0]) for y in range(img.shape[1])])

    ellipsoid = np.zeros(img.shape[0:2])
    ellipsoid_centroid = np.asarray([[0, 0]])
    ellipsoid_base_block = distance.cdist(pixel_coordinates, ellipsoid_centroid, metric=get_metric(h_weight, v_weight))
    ellipsoid_base_block = ellipsoid_base_block.reshape(img.shape[0:2])

    return ellipsoid_base_block


def get_metric(h_w, v_w):
    return lambda u, v: np.sqrt(((u * v_w - v * h_w) ** 2).sum())


def find_cc_centroids_areas(img):
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


def show_img(img, save=False):
    cv2.imshow('img', img)
    if save:
        cv2.imwrite("test.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# based on skimage
def cast_seam(img, energy_map):
    return transform.seam_carve(img, energy_map, mode='horizontal', num=1, border=0)


# try to find the seam based on the original and the cut image
def find_seam(ori_img, ori_cut_img):
    # resize the cut image
    # get the amount of rows to insert
    amount_rows = ori_img.shape[0] - ori_cut_img.shape[0]
    # add a line of zeros at the end to resize the image
    cut_img = np.append(ori_cut_img, np.zeros((amount_rows, ori_cut_img.shape[1], ori_cut_img.shape[2])), axis=0)

    img = np.copy(ori_img)
    img = np.sum(img, axis=2)
    img[img == 0] = 2

    # substract the seam carved image from the original one
    diff = img - np.sum(cut_img, axis=2)
    coords = [col.argmax() for col in (diff.transpose() != 0)]
    # coords is shit
    for idx, val in enumerate(coords):
        ori_img[val][idx][0] = 255

    return ori_img


def blur_energy_map_sc(img):
    horizontal_blur, _ = blur_image(img, filter_size=800)
    hori_verti_blur, _ = blur_image(horizontal_blur, filter_size=10, horizontal=False)
    hori_verti_blur = cv2.add(hori_verti_blur, img)
    img_energy_map, ori_energy_map = blur_image(hori_verti_blur, filter_size=100)
    show_energy = np.copy(img_energy_map)

    # cut_img = cast_seam(img, 300)
    # seam_img = find_seam(img, cut_img)

    for i in range(0, img.shape[0], 10):
        test = horizontal_seam(ori_energy_map, i)
        draw_seam(show_energy, test)

    show_img(show_energy)


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
    segment_textlines(input_loc='../data/test1.png')
    logging.info('Terminated')
