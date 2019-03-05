# Utils
import logging
import os
import time

import src.line_segmentation.preprocessing.energy_map
from src.line_segmentation.bin_algorithm import majority_voting
from src.line_segmentation.polygon_manager import polygon_to_string, get_polygons_from_lines, draw_polygons
from src.line_segmentation.preprocessing.load_image import prepare_image, load_image
from src.line_segmentation.preprocessing.preprocess import preprocess
from src.line_segmentation.seamcarving_algorithm import draw_seams, get_seams, post_process_seams, draw_seams_red
from src.line_segmentation.utils.XMLhandler import writePAGEfile
from src.line_segmentation.utils.graph_logger import GraphLogger
from src.line_segmentation.utils.util import create_folder_structure, save_img


#######################################################################################################################


def extract_textline(input_path, output_path, penalty_reduction=3000, seam_every_x_pxl=5,
                     testing=False, vertical=False, console_log=False):
    """
    Function to compute the text lines from a segmented image. This is the main routine where the magic happens
    """

    # -------------------------------
    start_whole = time.time()
    # -------------------------------

    ###############################################################################################
    # Load the image
    img = load_image(input_path)

    ###############################################################################################
    # Creating the folders and getting the new root folder
    root_output_path = create_folder_structure(input_path, output_path, (penalty_reduction, seam_every_x_pxl))

    # Init the logger with the logging path
    init_logger(root_output_path, console_log)

    # Init the graph logger
    GraphLogger.IMG_SHAPE = img.shape
    GraphLogger.ROOT_OUTPUT_PATH = root_output_path

    ###############################################################################################
    # Prepare image (filter only text, ...)
    img = prepare_image(img, testing=testing, cropping=False, vertical=vertical)
    # Pre-process the image
    save_img(img, path=os.path.join(root_output_path, 'preprocess', 'original.png'))
    img = preprocess(img)
    save_img(img, path=os.path.join(root_output_path, 'preprocess', 'after_preprocessing.png'))

    ###############################################################################################
    # Create the energy map
    energy_map, connected_components = src.line_segmentation.preprocessing.energy_map.create_energy_map(img,
                                                                                      blurring=False,
                                                                                      projection=False,
                                                                                      asymmetric=True)
    # Visualize the energy map as heatmap
    heatmap = src.line_segmentation.preprocessing.energy_map.create_heat_map_visualization(energy_map)
    save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_without_seams.png'))

    ###############################################################################################
    # Get the seams
    seams = get_seams(energy_map, penalty_reduction, seam_every_x_pxl)
    # Draw the seams on the heatmap
    draw_seams(heatmap, seams)
    save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_with_seams.png'))
    # Post-process the seams
    seams = post_process_seams(energy_map, seams)
    # Draw the seams on the heatmap
    draw_seams_red(heatmap, seams)
    save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_postprocessed_seams.png'))

    ###############################################################################################
    # Extract the bins
    lines = majority_voting(connected_components, seams)

    ###############################################################################################
    # Get polygons from lines
    polygons = get_polygons_from_lines(img, lines, connected_components, vertical)
    # Draw polygons overlay on original image
    save_img(draw_polygons(img.copy(), polygons, vertical), path=os.path.join(root_output_path, 'polygons_on_text.png'))

    ###############################################################################################
    # Write the results on the XML file
    writePAGEfile(os.path.join(root_output_path, 'polygons.xml'), polygon_to_string(polygons))

    ###############################################################################################
    # -------------------------------
    stop_whole = time.time()
    logging.info("finished after: {diff} s".format(diff=stop_whole - start_whole))
    # -------------------------------

    return


#######################################################################################################################
def init_logger(root_output_path, console_log):
    # create a logging format
    formatter = logging.Formatter(fmt='%(asctime)s %(filename)s:%(funcName)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    # get the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create and add file handler
    handler = logging.FileHandler(os.path.join(root_output_path, 'logs', 'extract_textline.log'))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if console_log:
        # create and add stderr handler
        stderr_handler = logging.StreamHandler()
        stderr_handler.formatter = formatter
        logger.addHandler(stderr_handler)

#######################################################################################################################
if __name__ == "__main__":

    extract_textline(input_path='./src/data/e-codices_fmb-cb-0055_0145v_max_gt.png',
                     output_path='./output',
                     seam_every_x_pxl=100,
                     penalty_reduction=6000,
                     testing=False,
                     console_log=True,
                     vertical=False)

    logging.info('Terminated')
