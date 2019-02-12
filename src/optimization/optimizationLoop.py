
import numpy as np
from sigopt import Connection
from src.image_segmentation.evaluate_algorithm import evaluate

INPUT_FOLDERS_PXL = ["/dataset/CB55/test-m"]  # , "/dataset/CSG18/test-m", "/dataset/CSG863/test-m)"]
INPUT_FOLDERS_XML = ["/dataset/CB55/test-page"]  # , "/dataset/CSG18/test-page", "/dataset/CSG863/test-page)"]
OUTPUT_FOLDER = "./output/"
NUM_CORES = 0
EVAL_TOOL = "./LineSegmentationEvaluator.jar"

def evaluate_metric(assignments):
    return evaluate(INPUT_FOLDERS_PXL, INPUT_FOLDERS_XML, OUTPUT_FOLDER, NUM_CORES, EVAL_TOOL,
                    assignments['penalty'], 1, assignments['seam_every_x_pxl'], 0)

if __name__ == '__main__':
    # Real Token
    # conn = Connection(client_token="YEQGRJZHNJMNHHZTDJIQKOXILQCSHZVFWWJIIWYNSWKQPGOA")
    # Dev Token
    conn = Connection(client_token="UQOOVYGGZNNDDFUAQQCCGMVNLVATTXDFKTXFXWIYUGRMJQHW") # DEV!!!!!!!!!!!!!
    conn.set_api_url("https://api.sigopt.com")

    experiment = conn.experiments().create(
        name="Line Segmentation - with merge",
        parameters=[
            dict(name="penalty", type="int", bounds=dict(min=1500, max=6000)),
            dict(name="seam_every_x_pxl", type="int", bounds=dict(min=1, max=10)),
        ],
        metrics=[dict(name="line IU")],
        observation_budget=50,
        parallel_bandwidth=1
    )

    # Run the Optimization Loop until she Observation Budget is exhausted
    while experiment.progress.observation_count < experiment.observation_budget:
        # Receive a suggestion
        suggestion = conn.experiments(experiment.id).suggestions().create()

        # Evaluate your metric
        value = evaluate_metric(suggestion.assignments)
        print(value)

        # Report an observation
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        # Update the experiment object
        experiment = conn.experiments(experiment.id).fetch()

    # Fetch the best configuration and explore your experiment
    best_assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments
    print("Best Assignments: " + best_assignments)
    print("Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")