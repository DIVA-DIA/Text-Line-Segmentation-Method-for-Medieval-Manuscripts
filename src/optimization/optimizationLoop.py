from sigopt import Connection

from src.line_segmentation.evaluation.evaluate_algorithm import evaluate

INPUT_FOLDERS_PXL = ["/dataset/Vietnamese/pxl_column_gt_diva"]
#INPUT_FOLDERS_PXL = ["/dataset/CB55/private-m" , "/dataset/CSG18/private-m", "/dataset/CSG863/private-m"]
GT_FOLDERS_XML = ["/dataset/Vietnamese/xml_column_gt"]
GT_FOLDERS_PXL = ["/dataset/Vietnamese/pxl_column_gt_diva"]
OUTPUT_FOLDER = "./output/"
NUM_CORES = 0
EVAL_TOOL = "./src/line_segmentation/evaluation/LineSegmentationEvaluator.jar"


def evaluate_metric(assignments):
    return evaluate(INPUT_FOLDERS_PXL, GT_FOLDERS_XML, GT_FOLDERS_PXL, OUTPUT_FOLDER, NUM_CORES, EVAL_TOOL,
                    assignments['penalty_reduction'], assignments['seam_every_x_pxl'], assignments['small_component_ratio'])


if __name__ == '__main__':
    # Real Token
    conn = Connection(client_token="NRDMOUKVYDGBWAGBNSOOHNDJBQAFAFHPMCDUURIRLGDIEFAB")
    # Dev Token
    # conn = Connection(client_token="JTAVMZCMWWUMPLEIMJUQGYMZRLMTXIYGXGKDAYGCVHTKOENK") # DEV!!!!!!!!!!!!!
    conn.set_api_url("https://api.sigopt.com")

    experiment = conn.experiments().create(
        name="HIP line segmentation v2",
        parameters=[
            dict(name="penalty_reduction", type="int", bounds=dict(min=3000, max=13000)),
            dict(name="seam_every_x_pxl", type="int", bounds=dict(min=10, max=50)),
            dict(name="small_component_ratio", type="double", bounds=dict(min=0.1, max=0.5)),
        ],
        metrics=[dict(name="line IU")],
        observation_budget=30,
        parallel_bandwidth=1,
        project='hip_line_segmentation'
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
    print("Best Assignments: " + str(best_assignments))
    print("Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")
