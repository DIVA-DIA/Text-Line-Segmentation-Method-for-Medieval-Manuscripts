# Utils
import logging
import os
import time

import cv2
import networkx as nx
import numpy as np
from skimage import measure

from src.line_segmentation.bin_algorithm import majority_voting
from src.line_segmentation.polygon_manager import polygon_to_string
from src.line_segmentation.seamcarving_algorithm import horizontal_seam, draw_seam
from src.line_segmentation.utils.XMLhandler import writePAGEfile
from src.line_segmentation.utils.graph_logger import GraphLogger
from src.line_segmentation.utils.graph_util import createTINgraph, print_graph_on_img
from src.line_segmentation.utils.util import create_folder_structure, save_img
from src.line_segmentation.preprocessing.load_image import prepare_image
from src.line_segmentation.preprocessing.energy_map import create_heat_map_visualization, prepare_energy, \
    create_energy_map


#######################################################################################################################


def extract_textline(input_loc, output_loc, show_seams=True, penalty=3000, nb_of_iterations=1, seam_every_x_pxl=5,
                     nb_of_lives=-1, too_small_pc=0.3, testing=False):
    """
    Function to compute the text lines from a segmented image. This is the main routine where the magic happens
    """

    # print("{}".format(read_max_textline_from_file('./../data/e-codices_fmb-cb-0055_0019r_max_gt.xml')))

    # -------------------------------
    start_whole = time.time()
    # -------------------------------

    # creating the folders and getting the new root folder
    root_output_path = create_folder_structure(input_loc, output_loc, (penalty, nb_of_iterations, seam_every_x_pxl, nb_of_lives))

    # inits the logger with the logging path
    init_logger(root_output_path)

    #############################################
    # Load the image
    img = cv2.imread(input_loc)

    #############################################
    # init the graph logger
    GraphLogger.IMG_SHAPE = img.shape
    GraphLogger.ROOT_OUTPUT_PATH = root_output_path
    #############################################

    # blow up image with the help of seams
    img, connected_components, last_seams = separate_textlines(img, root_output_path, penalty, show_seams,
                                                               testing, seam_every_x_pxl)

    nb_polygons = get_polygons(img, root_output_path, connected_components, last_seams, nb_of_lives, too_small_pc)

    logging.info("Amount of graphs: {amount}".format(amount=nb_polygons))

    # -------------------------------
    stop_whole = time.time()
    logging.info("finished after: {diff} s".format(diff=stop_whole - start_whole))
    # -------------------------------

    # validate
    return nb_polygons


def init_logger(root_output_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(os.path.join(root_output_path, 'logs', 'textline_extractor.log'))
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)


#######################################################################################################################


def separate_textlines(img, root_output_path, penalty, show_seams, testing, seam_every_x_pxl):
    """
    Contains the main loop. In each iteration it creates an energy map based on the given image CC and
    blows it up.
    """

    # -------------------------------
    start = time.time()
    # -------------------------------

    # Prepare image (filter only text, ...)
    img = prepare_image(img, testing=testing, cropping=False)

    # create the energy map
    ori_energy_map, cc = create_energy_map(img, blurring=False, projection=True, asymmetric=False)

    # visualize the energy map as heatmap
    heatmap = create_heat_map_visualization(ori_energy_map)

    save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_without_seams.png'), show=False)

    # list with all seams
    seams = []

    # left most column of the energy map
    left_column_energy_map = np.copy(ori_energy_map[:, 0])
    # right most column of the energy map
    right_column_energy_map = np.copy(ori_energy_map[:, -1])

    # show_img(ori_enegery_map)
    for seam_at in range(0, img.shape[0], seam_every_x_pxl):
        energy_map = prepare_energy(ori_energy_map, left_column_energy_map, right_column_energy_map, seam_at)

        seam = horizontal_seam(energy_map, penalty_div=penalty, bidirectional=True)
        seams.append(seam)
        if show_seams:
            draw_seam(heatmap, seam)

    last_seams = seams

    save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_with_seams_({}).png'.format(i)), show=False)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return img, cc, last_seams


def get_polygons(img, root_output_path, connected_components, last_seams, nb_of_lives, too_small_pc):
    # Mathias suggestion
    # compute the list of CC -> get them as parameter
    # for each pair of cc count how many times they were not separated
    # group as a single line all CCs which are most frequently together
    # -------------------------------
    start = time.time()
    # -------------------------------

    centroids = np.asarray([cc.centroid[0:2] for cc in connected_components[1]])
    centroids = centroids[np.argsort(centroids[:, 0]), :]

    # # a bit contains all centroids which belong to that line
    # lines = graph_to_point_lists(graphs)
    lines = majority_voting(centroids, last_seams)

    # Create a working copy of the image to draw the CC convex hull & so
    poly_img_text = np.asarray(img.copy(), dtype=np.float64)

    #############################################
    # Extract the contour of each CC
    nb_line = 0
    polygon_coords = []
    for line in lines:
        cc_coords = []
        graph_nodes = []
        polygon_img = np.zeros(img.shape)
        for c in line:
            cc = find_cc_from_centroid(c, connected_components[1])
            points = cc.coords[::3, 0:2]
            points = np.asarray([[point[1], point[0]]for point in points])
            cc_coords.append(points)
            # add the first countour point to the list s.t. the line will be connected
            graph_nodes.append([points[0][1], points[0][0]])

        # create graph
        overlay_graph = createTINgraph(graph_nodes)

        # create mst
        overlay_graph = nx.minimum_spanning_tree(overlay_graph)

        # overlay
        polygon_img = print_graph_on_img(polygon_img, [overlay_graph], color=(255, 255, 255), thickness=1)
        cv2.fillPoly(polygon_img, cc_coords, color=(255, 255, 255))

        # for cc_coord in cc_coords:
        #     # add cc areas to the image
        #     cv2.fillPoly(polygon_img, cc_coord, color=(255, 255, 255))

        # get contour points of the binary polygon image
        polygon_coords.append(measure.find_contours(polygon_img[:, :, 0], 254, fully_connected='high'))

        # take the biggest polygon
        contour = polygon_coords[nb_line][0]

        # print the polygon on the text
        cv2.polylines(poly_img_text, np.array([[[np.int(p[1]), np.int(p[0])] for p in contour]]), 1, color=(248, 24, 148))

        nb_line += 1

    # write into the xml file
    writePAGEfile(os.path.join(root_output_path, 'polygons.xml'), polygon_to_string(polygon_coords))

    save_img(poly_img_text, path=os.path.join(root_output_path, 'polygons_on_text.png'), show=False)

    # show_img(cc_img, show=True, save=True, name='polgyons_t1_fill.png')

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return nb_line


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


def find_cc_from_centroid(c, cc_properties):
#   c[0], c[1] = c[1], c[0]
    for cc in cc_properties:
        if cc.centroid[0] == c[0] and cc.centroid[1] == c[1]:
            return cc
    print("If this is printed, you might want to uncomment the line swapping the coordinates!")
    return None


def get_connected_components(img):
    cc_labels = measure.label(img[:, :, 1], background=0)
    cc_properties = measure.regionprops(cc_labels, cache=True)
    return cc_labels, cc_properties


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
    # extract_textline(input_loc='../data/A/19/e-codices_fmb-cb-0055_0019r_max_gt.png',
    #                  output_loc='../results/exp',
    #                  seam_every_x_pxl=5,
    #                  nb_of_lives=0,
    #                  testing=False)
    # extract_textline(input_loc='../data/test1.png',
    #                  output_loc='../../output',
    #                  seam_every_x_pxl=5,
    #                  nb_of_lives=0,
    #                  penalty=6000,
    #                  testing=True)
    extract_textline(input_loc='./../data/e-codices_fmb-cb-0055_0122v_max_output.png',
                     output_loc='./../../output',
                     seam_every_x_pxl=90,
                     penalty=5000,
                     testing=False)
    logging.info('Terminated')
