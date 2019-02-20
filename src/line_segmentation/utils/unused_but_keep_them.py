import logging
import time

import cv2
import numpy as np


def blow_up_image(image, seams):
    # -------------------------------
    start = time.time()
    # -------------------------------

    # new image
    new_image = []

    # get the new height of the image and the original one
    ori_height, _, _ = image.shape
    height = ori_height + len(seams)

    seams = np.array(seams)

    for i in range(0, image.shape[1]):
        col = np.copy(image[:, i])
        y_cords_seams = seams[:, i, 1]

        seam_nb = 0
        for y_seam in y_cords_seams:
            col = np.insert(col, y_seam + seam_nb, [0, 0, 0], axis=0)
            seam_nb += 1

        new_image.append(col)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return np.swapaxes(np.asarray(new_image), 0, 1), (((100 / ori_height) * height) - 100) / 100


def blur_image(img, save_name="blur_image.png", save=False, show=False, filter_size=1000, horizontal=True):
    # motion blur the image
    # generating the kernel
    kernel_motion_blur = np.zeros((filter_size, filter_size))
    if horizontal:
        kernel_motion_blur[int((filter_size - 1) / 2), :] = np.ones(filter_size)
    else:
        kernel_motion_blur[:, int((filter_size - 1) / 2)] = np.ones(filter_size)
    kernel_motion_blur = kernel_motion_blur / filter_size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)

    if save:
        cv2.imwrite(save_name, output)

    if show:
        cv2.imshow('image', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output  # , np.sum(output, axis=2)