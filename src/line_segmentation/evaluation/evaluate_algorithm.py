import argparse
import itertools
import os
import time
import re
import traceback

import numpy as np

from multiprocessing import Pool, cpu_count
from subprocess import Popen, PIPE, STDOUT
from src.line_segmentation.evaluation.overall_score import write_stats
from src.line_segmentation.line_segmentation import extract_textline
from src.pixel_segmentation.evaluation.apply_postprocess import apply_preprocess


def check_extension(filename, extension_list):
    return any(filename.endswith(extension) for extension in extension_list)


def get_file_list(dir, extension_list):
    list = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if check_extension(fname, extension_list):
                path = os.path.join(root, fname)
                list.append(path)
    list.sort()
    return list


def get_score(logs, token):
    for line in logs:
        line = str(line)
        if token in line:
            split = line.split('=')[1][0:8]
            split = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", split)
            return float(split[0])
    return None


def compute_for_all(input_img, gt_xml, gt_pxl, output_path, param_list, eval_tool):
    param_string = "_penalty_reduction_{}_seams_{}".format(
        param_list['penalty_reduction'],
        param_list['seam_every_x_pxl'])

    print("Starting: {} with {}".format(input_img, param_string))
    # Run the tool
    try:
        extract_textline(input_img, output_path, **param_list)
        print("Done: {} with {}".format(input_img, param_string))
    except:
        # for debugging
        print("Failed for some reason")
        return [None, traceback.format_exc(), param_list]

    line_extraction_root_folder = str(os.path.basename(input_img).split('.')[0] + param_string)

    # Run the JAR for PIXEL SEGMENTATION ################################################
    # print("Starting: (pixel) JAR {}".format(input_img))
    # apply_preprocess(input_image_path=input_img,
    #                  text_mask_path=os.path.join(output_path, line_extraction_root_folder, 'preprocess/after_preprocessing.png'),
    #                  output_path=os.path.join(output_path, 'postprocessd'))
    # p = Popen(['java', '-jar', './src/pixel_segmentation/evaluation/LayoutAnalysisEvaluator.jar',
    #            '-p', os.path.join(output_path, 'postprocessd', input_img.split('/')[-1]),
    #            '-gt', gt_pxl,
    #            '-dv'],
    #           stdout=PIPE, stderr=STDOUT)
    # print("Done: (pixel) JAR {} score={}".format(input_img,get_score( [line for line in p.stdout], "Mean IU (Jaccard index) =")))

    # Run the JAR for LINE SEGMENTATION ################################################
    # Check where is the path - Debugging only
    # p = Popen(['ls'], stdout=PIPE, stderr=STDOUT)
    # logs = [line for line in p.stdout]

    print("Starting: JAR {} with {}".format(input_img, param_string))
    p = Popen(['java', '-jar', eval_tool,
               '-igt', gt_pxl,
               '-xgt', gt_xml,
               '-xp', os.path.join(output_path, line_extraction_root_folder, 'polygons.xml'),
               '-csv'], stdout=PIPE, stderr=STDOUT)
    logs = [line for line in p.stdout]
    print("Done: JAR {} with {}".format(input_img, param_string))
    return [get_score(logs, "line IU ="), logs, param_list]


def evaluate(input_folders_pxl, gt_folders_xml, gt_folders_pxl, output_path, j, eval_tool,
             penalty_reduction, seam_every_x_pxl, **kwargs):

    # Select the number of threads
    if j == 0:
        pool = Pool(processes=cpu_count())
    else:
        pool = Pool(processes=j)

    # Get the list of all input images
    input_images = []
    for path in input_folders_pxl:
        input_images.extend(get_file_list(path, ['.png']))

    # Get the list of all GT XML
    gt_xml = []
    for path in gt_folders_xml:
        gt_xml.extend(get_file_list(path, ['.xml', '.XML']))

    # Get the list of all GT pxl
    gt_pxl = []
    for path in gt_folders_pxl:
        gt_pxl.extend(get_file_list(path, ['.png']))

    # Create output path for run
    tic = time.time()
    output_path = os.path.join(output_path, 'penalty_reduction_{}_seams_{}'.format(
        penalty_reduction,
        seam_every_x_pxl))

    if not os.path.exists(output_path):
        os.makedirs(os.path.join(output_path))

    # Debugging purposes only!
    #input_images = [input_images[0]]
    #gt_xml = [gt_xml[0]]
    #gt_pxl = [gt_pxl[0]]

    # For each file run
    param_list = dict(penalty_reduction=penalty_reduction, seam_every_x_pxl=seam_every_x_pxl)
    results = list(pool.starmap(compute_for_all, zip(input_images,
                                                gt_xml,
                                                gt_pxl,
                                                itertools.repeat(output_path),
                                                itertools.repeat(param_list),
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

    np.save(os.path.join(output_path, 'results.npy'), results)
    write_stats(output_path, errors)
    print('Total time taken: {:.2f}, avg_line_iu={}, nb_errors={}'.format(time.time() - tic, score, len(errors)))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search to identify best hyper-parameters for text line '
                                                 'segmentation.')
    # Path folders
    parser.add_argument('--input-folders-pxl', nargs='+', type=str,
                        required=True,
                        help='path to folders containing pixel-gt (e.g. /dataset/CB55/output-m /dataset/CSG18/output-m /dataset/CSG863/output-m)')

    parser.add_argument('--gt-folders-xml', nargs='+', type=str,
                        required=True,
                        help='path to folders containing xml-gt (e.g. /dataset/CB55/test-page /dataset/CSG18/test-page /dataset/CSG863/test-page)')
    parser.add_argument('--gt-folders-pxl', nargs='+', type=str,
                        required=True,
                        help='path to folders containing xml-gt (e.g. /dataset/CB55/test-m /dataset/CSG18/test-m /dataset/CSG863/test-m)')
    parser.add_argument('--output-path', metavar='DIR',
                        required=True,
                        help='path to store output files')

    # Method parameters
    parser.add_argument('--penalty-reduction', type=int,
                        required=True,
                        help='path to store output files')
    parser.add_argument('--seam-every-x-pxl', type=int,
                        required=True,
                        help='how many pixels between the seams')

    # Environment
    parser.add_argument('--eval-tool', metavar='DIR',
                        default='./src/line_segmentation/evaluation/LineSegmentationEvaluator.jar',
                        help='path to folder containing DIVA_Line_Segmentation_Evaluator')
    parser.add_argument('-j', type=int,
                        default=0,
                        help='number of thread to use for parallel search. If set to 0 #cores will be used instead')
    args = parser.parse_args()

    evaluate(**args.__dict__)
