import argparse
import itertools
import os
import time
from multiprocessing import Pool, cpu_count
from subprocess import Popen, PIPE, STDOUT

import numpy as np

from src.image_segmentation.overall_score import gather_stats
from src.image_segmentation.textline_extractor import extract_textline


def check_extension(filename, extension_list):
    return any(filename.endswith(extension) for extension in extension_list)


def get_file_list(dir, extension_list ):
    list = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if check_extension(fname, extension_list):
                path = os.path.join(root, fname)
                list.append(path)
    list.sort()
    return list


def get_score(logs):
    try:
        assert str(logs[-15]).split(' ')[-4:-2] == ['line', 'IU']
        score = float(str(logs[-15]).split(' ')[-1][:-3])
    except:
        print('Not line IU')
        score = 0.0
    return score


def compute_for_all(input_img, input_xml, output_path, param_list, eval_tool):
    param_string = "_penalty_{}_iterations_{}_seams_{}_lives_{}".format(
        param_list['penalty'],
        param_list['nb_of_iterations'],
        param_list['seam_every_x_pxl'],
        param_list['nb_of_lives'])

    print("Starting: {} with {}".format(input_img, param_string))
    # Run the tool
    try:
        extract_textline(input_img, output_path, **param_list)
    except:
        print("Failed for some reason")
        return [-1, [], param_list]

    # Run the JAR
    line_extraction_root_folder = str(os.path.basename(input_img).split('.')[0] + param_string)


    p = Popen(['java', '-jar', eval_tool,
               '-igt', input_img,
               '-xgt', input_xml,
               '-xp', os.path.join(output_path, line_extraction_root_folder, 'polygons.xml'),
               '-csv'], stdout=PIPE, stderr=STDOUT)
    logs = [line for line in p.stdout]
    score = get_score(logs)

    return [score, logs, param_list]


def evaluate(input_folders_pxl, input_folders_xml, output_path, j, eval_tool,
             penalty, nb_of_iterations, seam_every_x_pxl, nb_of_lives, **kwargs):
    # TODO make a new root folder which is the time so that we can make multiple runs for the same output folder
    # TODO give it as parameter



    # Select the number of threads
    if j == 0:
        pool = Pool(processes=cpu_count())
    else:
        pool = Pool(processes=j)

    # Get the list of all input images
    input_images = []
    for path in input_folders_pxl:
        input_images.extend(get_file_list(path, ['.jpg', '.png']))

    # Get the list of all input XML
    input_xml = []
    for path in input_folders_xml:
        input_xml.extend(get_file_list(path, ['.xml', '.XML']))

    # Create output path for run
    tic = time.time()
    current_time =  time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime())
    output_path = os.path.join(output_path, 'penalty_{}_seams_{}_lives_{}_iter_{}_t_{}'.format(
        penalty,
        seam_every_x_pxl,
        nb_of_lives,
        nb_of_iterations,
        current_time))

    if not os.path.exists(output_path):
        os.makedirs(os.path.join(output_path))

    # For each file run and log
    with open(os.path.join(output_path, 'logs.txt'), 'w') as f:
        # Iterate over the different dataset folders
        param_list = dict(penalty=penalty, seam_every_x_pxl=seam_every_x_pxl, nb_of_lives=nb_of_lives, nb_of_iterations=nb_of_iterations)
        results = list(pool.starmap(compute_for_all, zip(input_images,
                                                    input_xml,
                                                    itertools.repeat(output_path),
                                                    itertools.repeat(param_list),
                                                    itertools.repeat(eval_tool))))

        score = np.average([item[0] for item in results])
        log_message = 'penalty reduction: {} # of iterations: {} seam every x pixel: {} lives: {} score: {:.2f}\n'.format(
            penalty, nb_of_iterations, seam_every_x_pxl, nb_of_lives, score)
        f.write('Results for dataset {} \n'.format(os.path.dirname(os.path.dirname(image[0]))))
        f.write(log_message)
        print(log_message)
    pool.close()

    np.save('results.npy', results)
    gather_stats(output_path)
    print('Total time taken: {:.2f}'.format(time.time() - tic))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search to identify best hyper-parameters for text line '
                                                 'segmentation.')
    # Path folders
    parser.add_argument('--input-folders-pxl', nargs='+', type=str,
                        required=True,
                        help='path to folders containing pixel-gt (e.g. /dataset/CB55/test-m /dataset/CSG18/test-m /dataset/CSG863/test-m)')

    parser.add_argument('--input-folders-xml', nargs='+', type=str,
                        required=True,
                        help='path to folders containing xml-gt (e.g. /dataset/CB55/test-page /dataset/CSG18/test-page /dataset/CSG863/test-page)')

    parser.add_argument('--output-path', metavar='DIR',
                        required=True,
                        help='path to store output files')

    # Method parameters
    parser.add_argument('--penalty', type=int,
                        required=True,
                        help='path to store output files')
    parser.add_argument('--seam-every-x-pxl', type=int,
                        required=True,
                        help='how many pixels between the seams')
    parser.add_argument('--nb-of-lives', type=int,
                        required=True,
                        help='amount of lives an edge has')
    parser.add_argument('--nb-of-iterations', type=int,
                        default=1,
                        help='number of iterations')

    # Environment
    parser.add_argument('--eval_tool', metavar='DIR',
                        default='../data/LineSegmentationEvaluator.jar',
                        help='path to folder containing DIVA_Line_Segmentation_Evaluator')
    parser.add_argument('-j', type=int,
                        default=0,
                        help='number of thread to use for parallel search. If set to 0 #cores will be used instead')
    args = parser.parse_args()

    evaluate(**args.__dict__)
