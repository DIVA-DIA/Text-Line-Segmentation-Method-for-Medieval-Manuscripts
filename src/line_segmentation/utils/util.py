import shutil

import cv2
import os
import numpy as np


def create_folder_structure(input_file, output_path, params):
    """
    Creates the following folder structure:
    inputfilename_params
        - graph
        - histo

    :param input_file:
    :param output_path:
    :return:
    """
    fileName = os.path.basename(input_file).split('.')[0]

    # If the output_path does not exist (the output folder typically) then create it
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # basefolder
    basefolder_path = os.path.join(output_path, fileName + '_penalty_reduction_{}_seams_{}'.format(*params))
    # create basefolder
    if not os.path.exists(basefolder_path):
        os.mkdir(basefolder_path)
        # create energy maps folder
        os.mkdir(os.path.join(basefolder_path, 'energy_map'))
        # create histo folder
        os.mkdir(os.path.join(basefolder_path, 'logs'))
        # create preprocess folder
        os.mkdir(os.path.join(basefolder_path, 'preprocess'))

    return basefolder_path


def save_img(img, path='experiment.png', show=False):
    if show:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the image at the given path
    cv2.imwrite(path, img)


def calculate_asymmetric_distance(x, y, h_weight=1, v_weight=5):
    return [np.sqrt((((y[0] - x[0][0]) ** 2) * v_weight + ((y[1] - x[0][1]) ** 2) * h_weight)/ (h_weight+v_weight))]


def dict_to_string(dictionay):
    string = []
    for entry in dictionay.items():
        string.append('_'.join(entry))
    return '_'.join(string)
