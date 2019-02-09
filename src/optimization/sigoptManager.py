import sys
from sigopt import Connection


def delete_with_pattern(experiment_list, name):
    for n in experiment_list.data:
        if name in n.name:
            conn.experiments(n.id).delete()


def retrieve_id_by_name(experiment_list, name):
    retrieved = []
    for n in experiment_list.data:
        if name in n.name:
            retrieved.append(n.id)
    return retrieved

def print_with_pattern(experiment_list, name):
    dict = {}
    for n in experiment_list:
        if name in n.name:
            value = conn.experiments(n.id).best_assignments().fetch()
            if value.data:
                dict[n.name] = [value.data[0].value, n.progress.observation_count]
    for i in sorted(dict):
        print('{:100}'.format(i), "\t", dict[i])
    print("Found: " + str(len(dict)) + "\n\n")


if __name__ == '__main__':

    conn = Connection(client_token="YEQGRJZHNJMNHHZTDJIQKOXILQCSHZVFWWJIIWYNSWKQPGOA")

    # Fetch all experiments
    experiment_list = []
    for experiment in conn.experiments().fetch().iterate_pages():
        experiment_list.append(experiment)


    delete_with_pattern(experiment_list, "")


    print("Done!")

    #
    # EXPERIMENT_ID = retrieve_id_by_name(conn,"n__PureConv_32x32_/dataset/CIFAR10")
    #
    # if len(EXPERIMENT_ID) > 1:
    #     print("Experiments have duplicate names! Archive older ones before proceeding.")
    #     sys.exit(-1)
    # if not EXPERIMENT_ID:
    #     print("Experiments not found")
    #     sys.exit(-1)
    #
    # EXPERIMENT_ID = EXPERIMENT_ID[0]
    # best_assignments = conn.experiments(EXPERIMENT_ID).best_assignments().fetch()
    # lr = best_assignments.data[0].assignments['lr']
    # momentum = best_assignments.data[0].assignments['momentum']
