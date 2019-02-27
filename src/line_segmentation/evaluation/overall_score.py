import os
import csv
import numpy as np


def write_stats(path, errors):
    folders = list(os.walk(path))[0][1]
    stats = np.array([(_get_lines(folder, path), folder) for folder
                      in folders if os.path.exists(os.path.join(path, folder, 'results.csv'))])

    if list(stats):
        with open(os.path.join(path, 'logs.txt'), 'w') as f:
            avg_line_iu = np.average([np.float32(line[0][1][4]) for line in stats])
            print('Average lineUI: {}'.format(avg_line_iu))
            f.write("\n--------------------------------------------------\n\n")
            f.write('Average lineUI: {}'.format(avg_line_iu))

    with open(os.path.join(path, 'summary.csv'), 'w') as f:
        for i, line in enumerate(stats):
            if i == 0:
                f.write('filename,' + ','.join(line[0][0]) + '\n')
            f.write(line[1] + ',' + ','.join(line[0][1]) + '\n')

    if not errors:
        return

    with open(os.path.join(path, 'error_log.txt'), 'w') as f:
        for error in errors:
            f.writelines([line.decode('ascii') for line in error[1]])
            f.write("\n--------------------------------------------------\n\n")


def _get_lines(folder, path):
    csv_path = os.path.join(path, folder, 'polygons-results.csv')
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)


if __name__ == '__main__':
    write_stats(path='../results/line_seg_res_v1/')
