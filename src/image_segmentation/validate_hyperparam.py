import argparse
import itertools
import os
import time
from multiprocessing import Pool, cpu_count
from subprocess import Popen, PIPE, STDOUT

import numpy as np
from sklearn.model_selection import ParameterGrid

from src.image_segmentation.XMLhandler import read_max_textline_from_file
from src.image_segmentation.textline_extractor import extract_textline

# Specify the list of parameters to grid-search over.
# param_list = {'eps': [0.0061],
#               'min_samples': [3, 4],
#               'merge_ratio': [0.8]}
param_list = {
    'penalty': [3000],
    'nb_of_iterations': [1],
    'seam_every_x_pxl': [5],
    'nb_of_lives': [0]}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.png'])


def is_xml_file(filename):
    return any(filename.endswith(extension) for extension in ['.xml', '.XML'])


def get_list_images(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def get_list_xmls(dir):
    xmls = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_xml_file(fname):
                path = os.path.join(root, fname)
                xmls.append(path)
    return xmls


def get_score(logs):
    try:
        assert str(logs[-15]).split(' ')[-4:-2] == ['line', 'IU']
        score = float(str(logs[-15]).split(' ')[-1][:-3])
    except:
        print('Not line IU')
        score = 0.0
    return score


def compute_for_all(arg_container):
    input_pxl_img, input_path_xml, params, args = arg_container
    filename_without_ext = os.path.basename(input_pxl_img).split('.')[0]
    params['input_loc'] = input_pxl_img
    params['output_loc'] = args.output_path
    input_xml = os.path.join(input_path_xml, filename_without_ext + '.xml')

    output_loc = params['output_loc']
    try:
        num_lines = extract_textline(**params)
    except:
        print("Failed for some reason")
        score = 0.0
        logs = []
        return [params, score, logs]
    num_gt_lines = read_max_textline_from_file(input_xml)

    line_extraction_root_folder = filename_without_ext + '_penalty_{}_iterations_{}_seams_{}_lives_{}'.format(
                                                            params['penalty'],
                                                            params['nb_of_iterations'],
                                                            params['seam_every_x_pxl'],
                                                            params['nb_of_lives'])

    if True or num_gt_lines == num_lines:
        p = Popen(['java', '-jar', args.eval_tool,
                   '-igt', input_pxl_img,
                   '-xgt', input_xml,
                   '-xp', os.path.join(output_loc, line_extraction_root_folder, 'polygons.xml'),
                   '-csv'], stdout=PIPE, stderr=STDOUT)
        logs = [line for line in p.stdout]
        score = get_score(logs)
    else:
        print('Incorrect line count')
        score = 0.0
        logs = []
    return [params, score, logs]


def main(args):
    # TODO make a new root folder which is the time so that we can make multiple runs for the same output folder
    # TODO give it as parameter
    global param_list
    tic = time.time()
    input_images = []
    for path in args.input_folders_pxl:
        input_images.append(get_list_images(path))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    param_scores = []

    if args.j == 0:
        pool = Pool(cpu_count())
    else:
        pool = Pool(args.j)

    with open(os.path.join(args.output_path, 'logs.txt'), 'w') as f:
        for i, params in enumerate(ParameterGrid(param_list)):
            print('{} of {} parameters processed.'.format(i, len(ParameterGrid(param_list))))
            # iterate over the different dataset folders
            for j, dataset in enumerate(input_images):
                results = list(pool.map(compute_for_all, zip(dataset, itertools.repeat(args.input_folders_xml[j]),
                                                             itertools.repeat(params), itertools.repeat(args))))
                param_scores.append(results)
                score = np.average([item[1] for item in results])
                print('penalty reduction: {} # of iterations: {} seam every x pixel: {} lives: {} score: {:.2f}\n'.format(
                    params['penalty'],
                    params['nb_of_iterations'],
                    params['seam_every_x_pxl'],
                    params['nb_of_lives'],
                    score))
                f.write('Results for dataset {} \n'.format(os.path.dirname(os.path.dirname(dataset[0]))))
                f.write('penalty reduction: {} # of iterations: {} seam every x pixel: {} lives: {} score: {:.2f}\n'.format(
                    params['penalty'],
                    params['nb_of_iterations'],
                    params['seam_every_x_pxl'],
                    params['nb_of_lives'],
                    score))
                f.write('\n.........................................\n')
    pool.close()

    # score_matrix = []
    # with open('logs.txt', 'w') as f:
    #     for param_set in param_scores:
    #         eps = param_set[0][0]['eps']
    #         min_samples = param_set[0][0]['min_samples']
    #         merge_ratio = param_set[0][0]['merge_ratio']
    #         score = np.average([item[1] for item in param_set])
    #         f.write('eps: {} min_samples: {} merge_ratio: {} score: {:.2f}\n'.format(
    #             eps,
    #             min_samples,
    #             merge_ratio,
    #             score))
    #         score_matrix.append([eps, min_samples, merge_ratio, score])

    np.save('param_scores.npy', param_scores)
    print('Total time taken: {:.2f}'.format(time.time() - tic))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search to identify best hyper-parameters for text line '
                                                 'segmentation.')
    # Required arguments
    parser.add_argument('--input_folders_pxl', nargs='+', type=str,
                        help='path to folders containing pixel-gt')

    parser.add_argument('--input_folders_xml', nargs='+', type=str,
                        help='path to folders containing xml-gt')

    parser.add_argument('--output_path', metavar='DIR',
                        help='path to store output files')

    # optinal arguments
    parser.add_argument('--eval_tool', metavar='DIR',
                        default='../data/LineSegmentationEvaluator.jar',
                        help='path to folder containing DIVA_Line_Segmentation_Evaluator')
    parser.add_argument('-j', default=0, type=int,
                        help='number of thread to use for parallel search')
    args = parser.parse_args()

    main(args)
