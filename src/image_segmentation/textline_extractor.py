# Utils
import logging
import os
import time

import cv2
import sys
import numpy as np
import networkx as nx
from scipy.spatial import distance
from skimage import measure

#######################################################################################################################
from src.image_segmentation.utils.graph_logger import GraphLogger
from src.image_segmentation.utils.graph_util import createTINgraph, print_graph_on_img, cut_graph_with_seams, \
    graph_to_point_lists
from src.image_segmentation.utils.util import create_folder_structure, save_img
from src.image_segmentation.seamcarving import horizontal_seam, draw_seam
from src.image_segmentation.utils.XMLhandler import writePAGEfile


def extract_textline(input_loc, output_loc, show_seams=True, penalty=3000, nb_of_iterations=1, seam_every_x_pxl=5,
                     nb_of_lives=10, too_small_pc=0.3, testing=False):
    """
    Function to compute the text lines from a segmented image. This is the main routine where the magic happens
    :param input_loc: path to segmented image
    :param output_loc: path to the output folder
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
                                                               testing, seam_every_x_pxl, nb_of_iterations)

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


def separate_textlines(img, root_output_path, penalty, show_seams, testing, seam_every_x_pxl =5, nb_of_iterations=1):
    """
    Contains the main loop. In each iteration it creates an energy map based on the given image CC and
    blows it up.

    :param img:
    :param penalty:
    :param save_heatmap:
    :param show_seams:
    :param start_whole:
    :param show:
    :param nb_of_iterations:
    :return:
    """

    # -------------------------------
    start = time.time()
    # -------------------------------

    # list with all seams of the last iteration
    last_seams = []
    cc = []

    for i in range(nb_of_iterations):
        if i == 0:
            # Prepare image (filter only text, ...)
            img = prepare_image(img, testing=testing, cropping=False)

        # create the engergy map
        ori_energy_map, cc = create_energy_map(img, blurring=False, projection=True, asymmetric=False)

        # bidirectional energy map
        # bi_energy_map = build_seam_energy_map(ori_energy_map)

        # show_img((ori_enegery_map/max_en) * 255)
        # energy_map_representation = np.copy(ori_energy_map)

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

            seam = horizontal_seam(energy_map, penalty=True, penalty_div=penalty)
            seams.append(seam)
            if show_seams:
                draw_seam(heatmap, seam)

        if i != nb_of_iterations - 1:
            img, growth = blow_up_image(img, seams)
            penalty += penalty * growth
        else:
            last_seams = seams

        save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_with_seams.png'), show=False)

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
    # Lars
    # triangulate the CC
    # tranform into a graph
    graph = createTINgraph(centroids)

    # use the seams to cut them into graphs
    graphs = cut_graph_with_seams(graph, last_seams, too_small_pc)

    # iterate over all the sections of the seam as line and get from the quadtree the edges it could hit
    # if it hits a edge we delete this edge from the graph TODO give the edges 2 lives instead of just one
    GraphLogger.draw_graphs(img, graphs, name='cut_graph.png')
    graphs_as_point_lists = graph_to_point_lists(graphs)

    # Create a working copy of the image to draw the CC convex hull & so
    poly_img_text = np.asarray(img.copy(), dtype=np.float64)

    #############################################
    # Extract the contour of each CC
    # if the centroid is not on the area of the cc, replace centroid with random point on the area
    nb_line = 0
    polygon_coords = []
    for line in graphs_as_point_lists:
        cc_coords = []
        graph_nodes = []
        polygon_img = np.zeros(img.shape)
        for c in line:
            cc = find_cc_from_centroid(c, connected_components[1])
            points = cc.coords[::3, 0:2]
            points = np.asarray([[point[1], point[0]]for point in points])
            cc_coords.append(points)
            graph_nodes.append([points[0][1], points[0][0]])

            # hull = ConvexHull(points)
            # cc_polygons[l].append(points[hull.vertices][:, [1, 0]])
            # cc_hull_points.extend(points)
        # cc_polygons.append(np.asarray(cc_hull_points)[ConvexHull(cc_hull_points).vertices])

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


def prepare_image(img, testing, cropping=True):
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
        # Erase green just in case
        img[:, :, 1] = 0
        # Find and remove boundaries (set to bg)
        locations = np.where(img == 128)
        img[locations[0], locations[1]] = 0
        # Find regular text and text + decoration
        locations_text = np.where(img == 8)
        locations_text_comment = np.where(img == 12)
        # Wipe the image
        img[:, :, :] = 0
        # Set the text to be white
        img[locations_text[0], locations_text[1]] = 255
        img[locations_text_comment[0], locations_text_comment[1]] = 255

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return img


def blow_up_image(image, seams):
    # -------------------------------
    start = time.time()
    # -------------------------------

    # new image
    new_image = []

    # get the new height of the image and the original one
    ori_height, _, _ = image.shape
    height = ori_height + len(seams)

    seams = np.array(seams)

    for i in range(0, image.shape[1]):
        col = np.copy(image[:, i])
        y_cords_seams = seams[:, i, 1]

        seam_nb = 0
        for y_seam in y_cords_seams:
            col = np.insert(col, y_seam + seam_nb, [0, 0, 0], axis=0)
            seam_nb += 1

        new_image.append(col)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

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


def find_cc_from_centroid(c, cc_properties):
    c[0], c[1] = c[1], c[0]
    for cc in cc_properties:
        if (np.asarray(cc.centroid[0:2]) == c).all():
            return cc
    return None


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

    return output  # , np.sum(output, axis=2)


def calculate_asymmetric_distance(x, y, h_weight=1, v_weight=5):
    return [np.sqrt(((y[0] - x[0][0]) ** 2) * v_weight + ((y[1] - x[0][1]) ** 2) * h_weight)]


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
        template = np.array([calculate_asymmetric_distance(center_template, pxl) for pxl in pixel_coordinates]) \
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
        # normalize it between 0-1
        projection_profile = (projection_profile - np.min(projection_profile)) / (
                np.max(projection_profile) - np.min(projection_profile))
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


def create_projection_profile(map):
    # creating the horizontal projection profile
    projection_profile = np.sum(map, axis=1)
    return projection_profile


def find_cc_centroids_areas(img):
    # -------------------------------
    start = time.time()
    # -------------------------------
    #############################################
    # Find CC
    cc_labels, cc_properties = get_connected_components(img)

    amount_of_properties = 0

    while amount_of_properties != len(cc_properties):
        # for _ in range(2):
        amount_of_properties = len(cc_properties)
        #############################################
        # Cut all large components into smaller components
        img[:, :, 1] = cut_img(img[:, :, 1], cc_properties)

        # Re-find CC
        cc_labels, cc_properties = get_connected_components(img)
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


def get_connected_components(img):
    cc_labels = measure.label(img[:, :, 1], background=0)
    cc_properties = measure.regionprops(cc_labels, cache=True)
    return cc_labels, cc_properties


def polygon_to_string(polygons):
    # -------------------------------
    start = time.time()
    # -------------------------------
    strings = []
    for polygon in polygons:
        line_string = []
        for i, point in enumerate(polygon[0]):
            if i % 3 != 0:
                continue
            line_string.append("{},{}".format(int(point[1]), int(point[0])))
        strings.append(' '.join(line_string))

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------
    return strings


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
    #                  output_loc='../results/exp',
    #                  seam_every_x_pxl=5,
    #                  nb_of_lives=0,
    #                  penalty=6000,
    #                  testing=True)
    extract_textline(input_loc='./../data/test4.png',
                     output_loc='./../../output',
                     seam_every_x_pxl=10,
                     nb_of_lives=0,
                     penalty=2500,
                     testing=True)
    logging.info('Terminated')
