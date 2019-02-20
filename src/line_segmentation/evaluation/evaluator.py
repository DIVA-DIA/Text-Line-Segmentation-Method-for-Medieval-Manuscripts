import os
import re
import time
from subprocess import Popen, PIPE, STDOUT


def get_score(logs):
    for line in logs:
        line = str(line)
        if "line IU =" in line:
            line_ui = line.split('=')[1][0:8]
            line_ui = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line_ui)
            return float(line_ui[0])


def JavaTheFloorIsLava(input_img, input_xml, input_gt, eval_tool):
    print("RavaLaJava")
    tic = time.time()
    p = Popen(['java', '-jar', eval_tool,
               '-igt', input_img,
               '-xgt', input_gt,
               '-xp',  input_xml,
               '-csv'], stdout=PIPE, stderr=STDOUT)
    logs = [line for line in p.stdout]
    line_iu = get_score(logs)
    print('Done JAR time taken: {:.2f}, avg_line_iu={}'.format(time.time() - tic, line_iu))
    return line_iu


def JARJARiAMaPirate(input_img, input_xml, input_gt, output_path):
    print("Yarrr!!!!")
    return 10000


if __name__ == "__main__":
    NUM_CORES = 0
    EVAL_TOOL = "./src/evaluation/LineSegmentationEvaluator.jar"
    INPUT_PXL = "./src/evaluation/e-codices_csg-0863_005_max_gt.png"
    INPUT_XML = "./src/evaluation/polygons.xml"
    INPUT_GT  = "./src/evaluation/e-codices_csg-0863_005_max_gt.xml"

    a = JavaTheFloorIsLava(input_img=INPUT_PXL, input_xml=INPUT_XML, input_gt=INPUT_GT, eval_tool=EVAL_TOOL)
