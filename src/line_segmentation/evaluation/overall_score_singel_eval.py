import os
import csv
import numpy as np

CSV_FILE_NAME = 'polygons-results.csv'


def write_stats(path, errors):
    csv_paths = get_file_list(path, '.csv')
    stats = [_get_lines(path) for path in csv_paths]

    if list(stats):
        with open(os.path.join(path, 'logs.txt'), 'w') as f:

            avg_line_iu = np.average([np.float32(line[-1][4]) for line in stats])
            print('Average lineUI: {}'.format(avg_line_iu))
            f.write("\n--------------------------------------------------\n\n")
            f.write('Average lineUI: {}'.format(avg_line_iu))

        with open(os.path.join(path, 'summary.csv'), 'w') as f:
            for i, line in enumerate(stats):
                if i == 0:
                    f.write(','.join(line[0]) + '\n')
                f.write(','.join(line[-1]) + '\n')

    if not errors:
        return

    with open(os.path.join(path, 'error_log.txt'), 'w') as f:
        for error in errors:
            f.writelines([line.decode('ascii') for line in error[1]])
            f.write("\n--------------------------------------------------\n\n")


def _get_lines(path):
    with open(path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)


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


if __name__ == '__main__':
    write_stats(path='/dataset/Vietnamese/test_predictions_sys1/', errors=[])
