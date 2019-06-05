import logging
import time

import cv2
import networkx as nx
import numpy as np
from skimage import measure

from src.line_segmentation.utils.graph_util import createTINgraph, print_graph_on_img


def get_polygons_from_lines(img, lines, connected_components, vertical):

    # Extract the contour of each CC
    polygon_coords = []
    for i, line in enumerate(lines):
        cc_coords = []
        graph_nodes = []
        polygon_img = np.zeros(img.shape)
        for c in line:
            cc = find_cc_from_centroid(c, connected_components[1])
            points = cc.coords[::3, 0:2]
            points = np.asarray([[point[1], point[0]] for point in points])
            cc_coords.append(points)
            # add the first contour point to the list s.t. the line will be connected
            graph_nodes.append(find_graph_node(cc.coords, cc.centroid))

        # create graph
        overlay_graph = createTINgraph(np.array(list(set(tuple(p) for p in graph_nodes))))

        # create mst
        overlay_graph = nx.minimum_spanning_tree(overlay_graph)

        # overlay
        polygon_img = print_graph_on_img(polygon_img, [overlay_graph], color=(255, 255, 255), thickness=1)
        cv2.fillPoly(polygon_img, cc_coords, color=(255, 255, 255))

        # for cc_coord in cc_coords:
        #     # add cc areas to the image
        #     cv2.fillPoly(polygon_img, cc_coord, color=(255, 255, 255))

        if vertical:
            polygon_img = cv2.rotate(polygon_img, cv2.ROTATE_90_CLOCKWISE)

        # blurring the polygon image to close the gaps between the cut CCs
        filter_size_H = 5
        filter_size_V = 5
        kernel = np.ones((filter_size_V, filter_size_H)) / filter_size_H
        # Apply averaging filter
        polygon_img = cv2.filter2D(polygon_img, -1, kernel)

        # get contour points of the binary polygon image
        polygon_coords.append(measure.find_contours(polygon_img[:, :, 0], 5, fully_connected='high')[0])

    return polygon_coords


def find_graph_node(coords, centroid):
    # cast centroid coordinates into int
    centroid = np.asarray(centroid, dtype=int)

    # if centroid is in the coords we return the centroid
    if centroid in coords:
        return centroid

    # get the extreme points in coords on the same y as centroid
    # extrem_points = coords[np.where(coords[:, 1] == centroid[1])]

    return [coords[0][0], coords[0][1]]


def find_cc_from_centroid(c, cc_properties):
    #   c[0], c[1] = c[1], c[0]
    for cc in cc_properties:
        if cc.centroid[0] == c[0] and cc.centroid[1] == c[1]:
            return cc
    print("If this is printed, you might want to uncomment the line swapping the coordinates!")
    return None


def draw_polygons(image, polygons, vertical):
    if vertical:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    for polygon in polygons:
        cv2.polylines(image, np.array([[[np.int(p[1]), np.int(p[0])] for p in polygon]]), 1, color=(248, 24, 148), thickness=3)
    return image


def polygon_to_string(polygons):
    # -------------------------------
    start = time.time()
    # -------------------------------
    strings = []
    for polygon in polygons:
        line_string = []
        for i, point in enumerate(polygon):
            if i % 3 != 0:
                continue
            line_string.append("{},{}".format(int(point[1]), int(point[0])))
        strings.append(' '.join(line_string))

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------
    return strings