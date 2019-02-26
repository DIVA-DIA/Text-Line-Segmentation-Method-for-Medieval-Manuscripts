import os
import numpy as np
from src.line_segmentation.preprocessing.load_image import load_image
from src.line_segmentation.utils.util import save_img


def apply_preprocess(input_image_path, text_mask_path, output_path):

    # Load images from filesystem
    image = load_image(input_image_path)
    mask = load_image(text_mask_path)

    # Filter image with the text mask. Text bits are the value '8' so the 4th bit of the blue channel.
    locs = np.where((mask[:, :, 0] == 0) & (image[:, :, 0] != 1))
    for x, y in zip(locs[0], locs[1]):
        image[:, :, 0][x, y] &= ~(1 << 3)

    # Save the output to file
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_img(image, path=os.path.join(output_path, input_image_path.split('/')[-1]), show=False)
