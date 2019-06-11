import argparse
import itertools
import os
import re
import time
from multiprocessing import Pool, cpu_count
from subprocess import Popen, PIPE, STDOUT

import numpy as np

from src.line_segmentation.evaluation.evaluate_algorithm import get_file_list


def get_score(logs):
    for line in logs:
        line = str(line)
        if "Mean IU (Jaccard index) =" in line:
            pixel_iu = line.split('=')[1][0:8]
            pixel_iu = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", pixel_iu)
            return float(pixel_iu[0])
    return None


def compute_for_all(input_img, input_gt, output_path, eval_tool):

    # Check where is the path - Debugging only
    # p = Popen(['ls'], stdout=PIPE, stderr=STDOUT)
    # logs = [line for line in p.stdout]

    # Run the JAR
    print("Starting: JAR {}".format(input_img))
    p = Popen(['java', '-jar', eval_tool,
               '-p', input_img,
               '-gt', input_gt,
               '-dv'],
              stdout=PIPE, stderr=STDOUT)
    logs = [line for line in p.stdout]
    print("Done: JAR {}".format(input_img))
    return [get_score(logs), logs]


def evaluate(input_folders_pxl, input_folders_gt, output_path, eval_tool, j):

    # Select the number of threads
    if j == 0:
        pool = Pool(processes=cpu_count())
    else:
        pool = Pool(processes=j)

    # Get the list of all input images
    input_images = []
    for path in input_folders_pxl:
        input_images.extend(get_file_list(path, ['.png']))

    # Get the list of all input GTs
    input_gt = []
    for path in input_folders_gt:
        input_gt.extend(get_file_list(path, ['.png']))

    # Create output path for run
    if not os.path.exists(output_path):
        os.makedirs(os.path.join(output_path))

    # Debugging purposes only!
    # input_images = [input_images[4]]
    # input_gt = [input_gt[4]]
    #input_images = [input_images[0]]
    #input_gt = [input_gt[0]]
    # input_images = input_images[0:3]
    # input_gt = input_gt[0:3]

    tic = time.time()

    # For each file run
    results = list(pool.starmap(compute_for_all, zip(input_images,
                                                input_gt,
                                                itertools.repeat(output_path),
                                                itertools.repeat(eval_tool))))
    pool.close()
    print("Pool closed)")

    scores = []
    errors = []

    for item in results:
        if item[0] is not None:
            scores.append(item[0])
        else:
            errors.append(item)

    if list(scores):
        score = np.mean(scores)
    else:
        score = -1

    # TODO the save does not work for some reason, throws an error which does not happen in the LINE_IU scenario.
    #np.save(os.path.join(output_path, 'results.npy'), results)
    #write_stats(output_path, errors)
    print('Total time taken: {:.2f}, avg_pixel_iu={}, nb_errors={}'.format(time.time() - tic, score, len(errors)))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='May the odds be ever in your favor')
    # Path folders
    parser.add_argument('--input-folders-pxl', nargs='+', type=str,
                        required=True,
                        help='path to folders containing pixel-output (e.g. /dataset/CB55/output-m /dataset/CSG18/output-m /dataset/CSG863/output-m)')

    parser.add_argument('--input-folders-gt', nargs='+', type=str,
                        required=True,
                        help='path to folders containing pixel-gt (e.g. /dataset/CB55/test-m /dataset/CSG18/test-m /dataset/CSG863/test-m)')

    parser.add_argument('--output-path', metavar='DIR',
                        required=True,
                        help='path to store output files')

    # Environment
    parser.add_argument('--eval-tool', metavar='DIR',
                        default='./src/pixel_segmentation/evaluation/LayoutAnalysisEvaluator.jar',
                        help='path to folder containing DIVA_Line_Segmentation_Evaluator')
    parser.add_argument('-j', type=int,
                        default=0,
                        help='number of thread to use for parallel search. If set to 0 #cores will be used instead')
    args = parser.parse_args()

    evaluate(**args.__dict__)
