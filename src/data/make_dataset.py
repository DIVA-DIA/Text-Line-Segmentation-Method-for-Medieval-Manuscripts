import os
import argparse
import numpy as np
import cv2
import progressbar

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_list_images(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def process_gt(gt_img_path):
    gt_img = cv2.imread(gt_img_path)
    gt_img_proc = np.zeros((gt_img.shape[0], gt_img.shape[1]), dtype=np.uint8)

    # Find Border Pixels
    locs = np.where(gt_img[:,:,2] == 128)
    gt_img[locs[0], locs[1]] = 1

    # Find Background
    locs = np.where(gt_img[:, :, 0] == 1)
    gt_img_proc[locs[0], locs[1]] = 0

    # Find Text
    locs = np.where(gt_img[:, :, 0] == 8)
    gt_img_proc[locs[0], locs[1]] = 1


    # Find Decoration
    locs = np.where(gt_img[:, :, 0] == 4)
    gt_img_proc[locs[0], locs[1]] = 2


    # Find Comments
    locs = np.where(gt_img[:, :, 0] == 2)
    gt_img_proc[locs[0], locs[1]] = 3

    # Find Text+Decoration
    locs = np.where(gt_img[:, :, 0] == 12)
    gt_img_proc[locs[0], locs[1]] = 4

    # Find Text+Comments
    locs = np.where(gt_img[:, :, 0] == 10)
    gt_img_proc[locs[0], locs[1]] = 5


    # Find Text+Decoration
    locs = np.where(gt_img[:, :, 0] == 12)
    gt_img_proc[locs[0], locs[1]] = 6

    # Find Decoration+Comment
    locs = np.where(gt_img[:, :, 0] == 6)
    gt_img_proc[locs[0], locs[1]] = 7


    # Find Text+Decoration+Comment
    locs = np.where(gt_img[:, :, 0] == 14)
    gt_img_proc[locs[0], locs[1]] = 8


    return gt_img_proc


def get_filename_no_ext(name):
    return name.split('/')[-1].split('.')[0]


def make_dataset(data_folder, ground_truth, save_path, resize):
    data_list = get_list_images(data_folder)
    gt_list = get_list_images(ground_truth)
    try:
        assert len(data_list) == len(gt_list)
    except AssertionError:
        print("#Images in data_folder does not correspond to #images in ground_truth_folder")
    bar = progressbar.ProgressBar(max_value=len(data_list))
    train = []
    val = []
    test = []
    for i, (img_path, gt_img_path) in bar(enumerate(zip(sorted(data_list), sorted(gt_list)))):
        try:
            assert get_filename_no_ext(img_path) == get_filename_no_ext(gt_img_path)
        except AssertionError:
            print("Filenames do not correspond: {} \t {}".format(img_path, gt_img_path))

        category = img_path.split('/')[-2]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize is not None:
            img = cv2.resize(img, (0,0), fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
        gt_img = process_gt(gt_img_path)
        if resize is not None:
            gt_img = cv2.resize(gt_img, (0,0), fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
        if 8 in gt_img:
            locs = np.where(gt_img == 8)
            gt_img[locs[0], locs[1]] = 7
        if category == 'training':
            train.append([img, gt_img, img_path, gt_img_path])
        elif category == 'validation':
            val.append([img, gt_img, img_path, gt_img_path])
        elif category == 'test':
            test.append([img, gt_img, img_path, gt_img_path])
        else:
            print("Unknown category!")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'train.npy'), train)
    np.save(os.path.join(save_path, 'val.npy'), val)
    np.save(os.path.join(save_path, 'public-test.npy'), test)

def main(opts):
    make_dataset(opts.data_folder, opts.ground_truth, opts.save_path, opts.resize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help='Path to root')
    parser.add_argument('--ground_truth', help='Path to corresponding ground truth')
    parser.add_argument('--save_path', help='Path to folder where train/val/test splits will be saved.')
    parser.add_argument('--resize', type=float, help='Factor by which to resize images', default=None)
    args = parser.parse_args()
    main(args)
