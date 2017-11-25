import os
import argparse
import numpy as np
from subprocess import Popen, PIPE, STDOUT
from sklearn.model_selection import ParameterGrid

# Specify the list of parameters to grid-search over.
param_list = {'a': [1, 2, 3],
              'b': [2, 3, 4]}


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


def dummy_fn(input_image, output_loc, a, b):
    """
    Function to compute the text lines from a segmented image.
    :param input_image: path to segmented image
    :param output_loc: path to save generated PAGE XML
    :param a: param_a
    :param b: param_b
    :return: path to generated PAGE XML
    """
    return output_path


def get_score(logs):
    score = 1.0
    return score


def main(args):
    global param_list

    input_images = get_list_images(args.input_folder)
    jar_loc = os.path.join(args.eval_tool, 'out/artifacts/LineSegmentationEvaluator.jar')
    scores = []

    for img in input_images:
        for params in ParameterGrid(param_list):
            filename_without_ext = os.path.basename(img).split('.')[0]

            params['input_image'] = img
            params['output_loc'] = args.output_path

            output_loc = dummy_fn(**params)

            pixel_gt = os.path.join(args.gt_folder, filename_without_ext + '.png')
            page_gt = os.path.join(args.gt_folder, filename_without_ext + '.xml')

            p = Popen(['java', '-jar', jar_loc,
                       '-igt', pixel_gt,
                       '-xgt', page_gt,
                       '-xp', output_loc
                       ], stdout=PIPE, stderr=STDOUT)
            logs = [line for line in p.stdout]
            score = get_score(logs)
            scores.append(params, score, logs)

    np.save(scores)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search to identify best hyper-parameters for text line '
                                                 'segmentation.')
    # Required arguments
    parser.add_argument('input_folder', metavar='DIR',
                        help='path to folder containing input files')
    parser.add_argument('gt_folder', metavar='DIR',
                        help='path to folder containing pixel-GT and XML-GT files')
    # Optional argument
    parser.add_argument('--output_path', metavar='DIR',
                        default='/tmp', help='path to store temporary output files')
    parser.add_argument('--eval_tool', metavar='DIR',
                        default='./DIVA_Line_Segmentation_Evaluator',
                        help='path to folder containing DIVA_Line_Segmentation_Evaluator')
    args = parser.parse_args()

    main(args)
