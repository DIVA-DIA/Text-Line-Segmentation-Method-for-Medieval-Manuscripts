"""
"""

import cv2
# Utils
import logging
import numpy as np
import os
from XMLhandler import writePAGEfile
from scipy.spatial import ConvexHull
from skimage import measure
from sklearn.cluster import DBSCAN


#######################################################################################################################

def segment_textlines(input_loc, output_loc, eps=0.0061, min_samples=4, simplified=False, visualize=False):
    """
    Function to compute the text lines from a segmented image. This is the main routine where the magic happens
    :param input_loc: path to segmented image
    :param output_loc: path to save generated PAGE XML
    :param a: param_a
    :param b: param_b
    """

    #print("{}".format(read_max_textline_from_file('./../data/e-codices_fmb-cb-0055_0019r_max_gt.xml')))

    #############################################
    # Load the image
    img = cv2.imread(input_loc)

    # Prepare image (filter only text, ...)
    img = prepare_image(img)

    #############################################
    # Find CC
    cc_labels = measure.label(img[:,:,1], background=0)
    cc_properties = measure.regionprops(cc_labels, cache=True)

    # Collect CC centroids
    all_centroids = []
    for cc in cc_properties:
        all_centroids.append(cc.centroid[0:2])
    all_centroids = np.asarray(all_centroids)
    all_centroids = all_centroids[np.argsort(all_centroids[:, 0]), :]

    # Collect CC sizes
    area = []
    for cc in cc_properties:
        area.append(cc.area)
    area = np.asarray(area)

    # Split centroids who are too big
    for i,c in enumerate(all_centroids):
        if area[i] > 3 * np.mean(area):
            cc = find_cc_from_centroid(c, cc_properties)
            if abs(cc.orientation) < 3.14/4:

                # On their location
                cv2.circle(img, tuple(reversed(np.round(c).astype(np.int))), radius=10,
                           color=(0, 255, 255), thickness=20, lineType=1, shift=0)


    # Draw centroids [ALL]
    for c in all_centroids:
        # On their location
        cv2.circle(img, tuple(reversed(np.round(c).astype(np.int))), radius=5,
                   color=(200, 0, 0), thickness=10, lineType=1, shift=0)
        # On the side
        cv2.circle(img, tuple([50, np.round(c[0]).astype(np.int)]), radius=2,
                   color=(200, 0, 0), thickness=2, lineType=1, shift=0)

    #############################################
    # Discard outliers & sort
    no_outliers = detect_outliers(all_centroids[:, 0], area)
    centroids = all_centroids[no_outliers, :]
    filtered_area = area[no_outliers]
    filtered_area = filtered_area[np.argsort(centroids[:, 0])]
    centroids = centroids[np.argsort(centroids[:, 0]), :]

    # Draw centroids [NO_OUTLIERS]
    for i, c in enumerate(centroids):
        # On their location
        cv2.circle(img, tuple(reversed(np.round(c).astype(np.int))), radius=5,
                   color=(0, 200, 0), thickness=10, lineType=1, shift=0)
        # On the side
        tmp = (filtered_area - np.min(filtered_area)) / (np.max(filtered_area) - np.min(filtered_area))

        cv2.circle(img, tuple([np.round(100+tmp[i]*50).astype(np.int) , np.round(c[0]).astype(np.int)]), radius=2,
                   color=(0, 200, 0), thickness=2, lineType=1, shift=0)

    #############################################
    # Cluster the points and draw the clusters
    # TODO add the area as 2nd dimensions instead of zeros?
    centroids_after_clustering, labels = cluster(img, centroids, filtered_area, eps, min_samples)
    clusters_lines = draw_clusters(img, centroids_after_clustering, labels)

    #############################################
    # Compute line width
    lines_width = []

    points = points_in_line(cc_properties, [centroids_after_clustering[0]])
    top_line = np.round(np.min(points[:, 0])).astype(np.int)
    lines_width.append(clusters_lines[0]-top_line)

    for i in range(0, len(clusters_lines)-1):
        lines_width.append(clusters_lines[i+1] - clusters_lines[i])

    points = points_in_line(cc_properties, [centroids_after_clustering[-1]])
    bottom_line = np.round(np.max(points[:, 0])).astype(np.int)
    lines_width.append(bottom_line-clusters_lines[-1])

    # Detect lines too small
    lines_too_small = lines_width < 0.7 * np.mean(lines_width)

    # Select lines to be removed
    index = []
    if lines_too_small[0]:
        index.append(0)

    if lines_too_small[-1]:
        index.append(len(clusters_lines))
    """
    for i in range(1, len(lines_too_small)-1):    
        # Merge all small lines
        if lines_too_small[i]:
            # Merge with your smallest neighbor
            if lines_width[i-1] < lines_width[i+1]:
                index.append(i-1)
            else:
                index.append(i)
                lines_too_small[i+1] = False
    """
    for i in range(0, len(lines_too_small) - 1):
        # Merge pairs
        if lines_too_small[i] and lines_too_small[i + 1]:
            index.append(i)
            lines_too_small[i] = False
            lines_too_small[i + 1] = False

    # Merge them by removing the lines
    clusters_lines = np.delete(clusters_lines, index)

    # Print num clusters
    print("C:{}".format(len(clusters_lines)+1))

    # Draw centroids [AFTER CLUSTERING]
    for i, c in enumerate(centroids_after_clustering):
        if no_outliers[i]:
            # On their location
            cv2.circle(img, tuple(reversed(np.round(c).astype(np.int))), radius=5,
                       color=(0, 0, 200), thickness=10, lineType=1, shift=0)

    # Get the polygons around text
    boxes = []
    if simplified:
        boxes = draw_boxes(all_centroids, cc_properties, clusters_lines)
    else:
        boxes = draw_contour(all_centroids, cc_properties, centroids_after_clustering, clusters_lines, img)

    # Save bounding box for each row as PAGE file
    writePAGEfile(output_loc, textLines=boxes)

    if visualize:
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.resizeWindow('image', 1200, 900)
        cv2.moveWindow('image', 200, 25)

        # Hold on
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return the number of clusters
    return len(clusters_lines)+1



#######################################################################################################################


def prepare_image(img):
    """
    img[:, :, 0] = 0
    img[:, :, 2] = 0
    locations = np.where(img == 127)
    img[:, :, 1] = 0
    img[locations[0], locations[1]] = 255
    """
    # Erase green (if any, but shouldn't have values here)
    img[:, :, 1] = 0
    # Find and remove boundaries (set to bg)
    boundaries = np.where(img == 128)
    img[boundaries[0], boundaries[1]] = 0
    # Find regular text and make it white
    locations = np.where(img == 8)
    img[locations[0], locations[1]] = 128
    # Find text+decoration and make it white
    locations = np.where(img == 12)
    img[locations[0], locations[1]] = 128
    # Erase red & blue (so we get rid of everything else, only green will be set)
    img[:, :, 0] = 0
    img[:, :, 2] = 0

    return img


def detect_outliers(centroids, area):
    big_enough = area > 0.4 * np.mean(area)
    #small_enough = area < 3 * np.mean(area)
    small_enough = area > 0
    no_y = abs(centroids - np.mean(centroids)) < 2 * np.std(centroids)
    no_outliers = [x & y & z for (x, y, z) in zip(big_enough, small_enough, no_y)]
    return no_outliers


def cluster(img, centroids, area, eps, min_samples):
    # Attempt clustering with DBSCAN
    X = centroids[:, 0]
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    area = (area - np.min(area)) / (np.max(area) - np.min(area))
    tmp = np.zeros((X.shape[0], 2))
    #tmp[:, 1] = area
    tmp[:, 0] = X
    X = tmp

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


def draw_boxes(all_centroids, cc_properties, clusters_lines):
    boxes = []

    # Separate the centroids in cluster bins WITH ALL CENTROIDS
    clusters_centroids = separate_in_bins(all_centroids, clusters_lines)
    # Compute upper bound of the text area and add FIRST line
    points = points_in_line(cc_properties, clusters_centroids[0], fast=True)
    top_line = np.round(np.min(points[:, 0])).astype(np.int)
    left = np.round(np.min(points[:, 1])).astype(np.int)
    right = np.round(np.max(points[:, 1])).astype(np.int)
    boxes.append("{},{} {},{} {},{} {},{}".format(right, top_line,
                                                  left, top_line,
                                                  left, np.floor(clusters_lines[0]).astype(np.int),
                                                  right, np.floor(clusters_lines[0]).astype(np.int)))
    # Add all intermediate lines (not the first/last ones)
    for i, line in enumerate(clusters_centroids[1:-1]):
        points = points_in_line(cc_properties, line, fast=True)

        left = np.round(np.min(points[:, 1])).astype(np.int)
        top = np.floor(clusters_lines[i]).astype(np.int)
        right = np.round(np.max(points[:, 1])).astype(np.int)
        bottom = np.ceil(clusters_lines[i + 1]).astype(np.int)
        boxes.append("{},{} {},{} {},{} {},{}".format(right, top, left, top, left, bottom, right, bottom))

    # Compute lower bound of the text area and add LAST line
    points = points_in_line(cc_properties, clusters_centroids[-1], fast=True)
    bottom_line = np.round(np.max(points[:, 0])).astype(np.int)
    left = np.round(np.min(points[:, 1])).astype(np.int)
    right = np.round(np.max(points[:, 1])).astype(np.int)
    boxes.append("{},{} {},{} {},{} {},{}".format(right, np.ceil(clusters_lines[-1]).astype(np.int),
                                                  left, np.ceil(clusters_lines[-1]).astype(np.int),
                                                  left, bottom_line,
                                                  right, bottom_line))
    return boxes


def draw_contour(all_centroids, cc_properties, centroids_after_clustering, clusters_lines, img):
    boxes = []

    # Separate ALL the centroids in cluster bins
    clusters_centroids = separate_in_bins(all_centroids, clusters_lines)

    #############################################
    # Compute line medians
    lines_median = []
    points = points_in_line(cc_properties, [centroids_after_clustering[0]])
    top_line = np.round(np.min(points[:, 0])).astype(np.int)
    lines_median.append(np.round((clusters_lines[0] + top_line) / 2.0).astype(np.int))
    for i in range(0, len(clusters_lines) - 1):
        lines_median.append(np.round((clusters_lines[i + 1] + clusters_lines[i]) / 2.0).astype(np.int))
    points = points_in_line(cc_properties, [centroids_after_clustering[-1]])
    bottom_line = np.round(np.max(points[:, 0])).astype(np.int)
    lines_median.append(np.round((bottom_line + clusters_lines[-1]) / 2.0).astype(np.int))

    #############################################
    # Extract the contour of each CC
    cc_polygons = []
    cc_img = []
    # For all lines
    for l, line in enumerate(clusters_centroids):
        # Create a working copy of the image to draw the CC convex hull & so
        cc_img.append(np.zeros(img.shape[0:2] + (3,), dtype=np.uint8))

        # For each cc in the line
        cc_polygons.append([])
        for i, c in enumerate(line):
            # Draw a line from most-left to the first centroid (might be needed if line starts with a C)
            #cv2.line(cc_img, tuple([0, np.round(lines_median[l]).astype(np.int)]),
            #         tuple([4000, np.round(lines_median[l]).astype(np.int)]),
            #         color=(255, 255, 255), thickness=10, lineType=8, shift=0)

            # Draw the line between the centroids
            if i < len(line) - 1:
                # On the image
                #cv2.line(img, tuple([np.round(line[i][1]).astype(np.int), np.round(line[i][0]).astype(np.int)]),
                #         tuple([np.round(line[i + 1][1]).astype(np.int), np.round(line[i + 1][0]).astype(np.int)]),
                #         color=(255, 127, 0), thickness=10, lineType=8, shift=0)
                # Draw it on the working copy
                cv2.line(cc_img[l], tuple([np.round(line[i][1]).astype(np.int), np.round(line[i][0]).astype(np.int)]),
                         tuple([np.round(line[i + 1][1]).astype(np.int), np.round(line[i + 1][0]).astype(np.int)]),
                         color=(255, 255, 255), thickness=10, lineType=8, shift=0)

            # Retrieve the cc and its points
            cc = find_cc_from_centroid(c, cc_properties)
            points = cc.coords[:, 0:2]
            if (len(points) < 100):
                continue

            # Draw the points on the working copy
            cc_img[l][points[:, 0], points[:, 1]] = 255

        # Print workig copy
        """
        cv2.namedWindow('canvas', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('canvas', cc_img[l])
        cv2.resizeWindow('image', 1200, 900)
        cv2.moveWindow('image', 200, 25)

        # Hold on
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
    #############################################
    # Querying the new global CC on the working copies
    for l, line in enumerate(clusters_centroids):
        cc_img[l] = cv2.cvtColor(cc_img[l], cv2.COLOR_BGR2GRAY)
        text_line_contour = cv2.findContours(cc_img[l], mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_L1)
        text_line_contour = text_line_contour[1][0]
        # Draw contour on image
        cv2.polylines(img, [text_line_contour], isClosed=True, thickness=5, color=(0, 255, 255))

        boxes.append("")
        for point in text_line_contour[::3]:
            boxes[l] += "{},{} ".format(point[0][1], point[0][0])

    return boxes


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
    segment_textlines(input_loc='./../data/e-codices_fmb-cb-0055_0019r_max_gt.png',
                      output_loc="./../data/testfile.txt",
                      visualize=True,
                      simplified=False)
    logging.info('Terminated')
