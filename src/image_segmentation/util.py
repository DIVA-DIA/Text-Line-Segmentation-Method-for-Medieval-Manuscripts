import os


def create_folder_structure(input_file, output_path):
    """
    Creates the follwoing folder structure:
    inputfilename_params
        - graph
        - histo

    :param input_file:
    :param output_path:
    :return:
    """
    fileName = os.path.basename(input_file).split('.')[0]

    # basefolder
    basefolder_path = os.path.join(output_path, fileName)
    # create basefolder
    os.mkdir(basefolder_path)
    # create graph folder
    os.mkdir(os.path.join(basefolder_path, 'graph'))
    # create energy maps folder
    os.mkdir(os.path.join(basefolder_path, 'energy_map'))
    # create histo folder
    os.mkdir(os.path.join(basefolder_path, 'histo'))

    return basefolder_path
