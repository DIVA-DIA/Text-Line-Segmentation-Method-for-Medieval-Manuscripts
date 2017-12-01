import argparse
import itertools
import os
import time
from multiprocessing import Pool
from subprocess import Popen, PIPE, STDOUT

import numpy as np
from sklearn.model_selection import ParameterGrid

from XMLhandler import read_max_textline_from_file
from textline_extractor import segment_textlines

# Specify the list of parameters to grid-search over.
param_list = {'eps': [0.0061],
              'min_samples': [3, 4],
              'merge_ratio': [0.8]}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.png'])


def get_list_images(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def get_score(logs):
    try:
        assert str(logs[-13]).split(' ')[-4:-2] == ['line', 'IU']
        score = float(str(logs[-13]).split(' ')[-1][:-3])
    except:
        print('Not line IU')
        score = 0.0
    return score


def compute_for_all(arg_container):
    input_loc, params, args = arg_container
    filename_without_ext = os.path.basename(input_loc).split('.')[0]
    params['input_loc'] = input_loc
    params['output_loc'] = os.path.join(args.output_path,
                                        filename_without_ext + '_eps_{}_min_samples_{}_merge_ration_{}.xml'.format(
                                            params['eps'],
                                            params['min_samples'],
                                            params['merge_ratio'],))
    output_loc = params['output_loc']
    try:
        num_lines = segment_textlines(**params)
    except:
        print("Failed for some reason")
        score = 0.0
        logs = []
        return [params, score, logs]
    pixel_gt = os.path.join(args.gt_folder, filename_without_ext + '.png')
    page_gt = os.path.join(args.gt_folder, filename_without_ext + '.xml')
    num_gt_lines = read_max_textline_from_file(page_gt)

    if True or num_gt_lines == num_lines:
        p = Popen(['java', '-jar', args.eval_tool,
                   '-igt', pixel_gt,
                   '-xgt', page_gt,
                   '-xp', output_loc
                   ], stdout=PIPE, stderr=STDOUT)
        logs = [line for line in p.stdout]
        score = get_score(logs)
    else:
        print('Incorrect line count')
        score = 0.0
        logs = []
    return [params, score, logs]


def main(args):
    global param_list
    tic = time.time()
    input_images = get_list_images(args.input_folder)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    param_scores = []

    pool = Pool(args.j)

    with open('logs.txt', 'w') as f:
        for i, params in enumerate(ParameterGrid(param_list)):
            print('{} of {} parameters processed.'.format(i, len(ParameterGrid(param_list))))
            results = list(pool.map(compute_for_all, zip(input_images, itertools.repeat(params), itertools.repeat(args))))
            param_scores.append(results)
            score = np.average([item[1] for item in results])
            print('eps: {} min_samples: {} merge_ratio: {} score: {:.2f}\n'.format(
                params['eps'],
                params['min_samples'],
                params['merge_ratio'],
                score))
            f.write('eps: {} min_samples: {} merge_ratio: {} score: {:.2f}\n'.format(
                params['eps'],
                params['min_samples'],
                params['merge_ratio'],
                score))
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
    parser.add_argument('--input_folder', metavar='DIR',
                        help='path to folder containing input files')
    parser.add_argument('--gt_folder', metavar='DIR',
                        help='path to folder containing pixel-GT and XML-GT files')
    # Optional argument
    parser.add_argument('--output_path', metavar='DIR',
                        default='/tmp', help='path to store temporary output files')
    parser.add_argument('--eval_tool', metavar='DIR',
                        default='./DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar',
                        help='path to folder containing DIVA_Line_Segmentation_Evaluator')
    parser.add_argument('-j', default=8, type=int,
                        help='number of thread to use for parallel search')
    args = parser.parse_args()

    main(args)
