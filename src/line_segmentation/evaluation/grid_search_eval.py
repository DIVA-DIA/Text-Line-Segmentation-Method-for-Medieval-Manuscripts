import time

from sklearn.model_selection import ParameterGrid
from src.line_segmentation.evaluation.evaluate_algorithm import evaluate

PARAM_LIST = {'seam_every_x_pxl': list(range(20, 110, 10)),
              'penalty_reduction': list([13000])}
OUTPUT_FOLDER = "./output/"
NUM_CORES = 0
EVAL_TOOL = "./src/line_segmentation/evaluation/LineSegmentationEvaluator.jar"

INPUT_FOLDERS_PXL = [
    "/dataset/CB55/private-m", "/dataset/CSG18/output-processed-m", "/dataset/CSG863/output-processed-m"]
GT_FOLDERS_XML = [
    "/dataset/CB55/private-page", "/dataset/CSG18/private-page", "/dataset/CSG863/private-page"]
GT_FOLDERS_PXL = [
    "/dataset/CB55/private-m", "/dataset/CSG18/private-m", "/dataset/CSG863/private-m"]

if __name__ == '__main__':

    results = []
    with open('log.txt', 'w') as file:
        for params in ParameterGrid(PARAM_LIST):
            start = time.time()
            print("start: {}".format(time.localtime()))
            score = evaluate(INPUT_FOLDERS_PXL, GT_FOLDERS_XML, GT_FOLDERS_PXL, OUTPUT_FOLDER, NUM_CORES, EVAL_TOOL,
                             params['penalty_reduction'], params['seam_every_x_pxl'])
            results.append(dict(score=score, params=params))
            print("end: {}".format(start - time.time()))
            file.write("score: {}   penalty: {} seam_every: {}\n".format(score, params['penalty_reduction'], params['seam_every_x_pxl']))
            # delete all files from the this run

    print(results)
