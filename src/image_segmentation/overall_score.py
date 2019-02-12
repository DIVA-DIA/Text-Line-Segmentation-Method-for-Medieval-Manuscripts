import os
import csv
import numpy as np


def gather_stats(path):
    folders = list(os.walk(path))[0][1]
    stats = np.array([(_get_lines(folder, path), folder) for folder
                      in folders if os.path.exists(os.path.join(path, folder, 'results.csv'))])

    with open(os.path.join(path, 'logs.txt'), 'w') as f:
        avg_line_iu = np.average([np.float32(line[0][1][4]) for line in stats])
        print('Average lineUI: {}'.format(avg_line_iu))
        f.write('\n\n--------------------\n')
        f.write('Average lineUI: {}'.format(avg_line_iu))

    with open(os.path.join(path, 'summary.csv'), 'w') as f:
        for i, line in enumerate(stats):
            if i == 0:
                f.write('filename,' + ','.join(line[0][0]) + '\n')
            f.write(line[1] + ',' + ','.join(line[0][1]) + '\n')

    return avg_line_iu

def _get_lines(folder, path):
    csv_path = os.path.join(path, folder, 'results.csv')
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)


if __name__ == '__main__':
    gather_stats(path='../results/line_seg_res_v1/')
