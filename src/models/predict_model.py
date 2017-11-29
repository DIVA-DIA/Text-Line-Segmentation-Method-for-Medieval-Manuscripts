import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
from sklearn.preprocessing import normalize
from PIL import Image
import cv2
from skimage.util import pad


parser = argparse.ArgumentParser(description='Layout Analysis ICDAR2017')

parser.add_argument('input_folder', metavar='DIR',
                    help='path to input folder (CB55, CS18, CS863)')

parser.add_argument('output_folder', metavar='DIR',
                    help='path to output folder')

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--weights', default='', type=str, metavar='PATH',
                    help='path to model weights (default: none)')

parser.add_argument('--resize', type=float,
                    help='factor by which to resize images', default=1.0)


class ImageLoader():
    def __init__(self, data_path, resize, transform=None):
        self.img_path = data_path
        self.transform = transform
        self.img = cv2.imread(self.img_path)
        self.orig_x = self.img.shape[0]
        self.orig_y = self.img.shape[1]
        self.img = cv2.resize(self.img, (0, 0), fx=resize, fy=resize,
                              interpolation=cv2.INTER_NEAREST)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.x = self.img.shape[0]
        self.y = self.img.shape[1]
        self.len = self.x * self.y

    def crop_with_padding(self, img, x, y, SIZE):
        def simplify(num):
            if num < 0:
                return abs(num)
            else:
                return 0

        SIZE = int(SIZE / 2)
        glx, gly, grx, gry = x - SIZE, y - SIZE, img.shape[0] - (x + SIZE), img.shape[1] - (y + SIZE)
        # print('{} {} {} {}'.format(glx, gly, grx, gry))

        lx = max(0, x - SIZE)
        ly = max(0, y - SIZE)
        rx = min(img.shape[0], x + SIZE)
        ry = min(img.shape[1], y + SIZE)



        new_img = img[lx:rx, ly:ry]

        if len(img.shape) == 3:
            new_img = pad(new_img, ((simplify(glx), simplify(grx)), (simplify(gly), simplify(gry)), (0, 0)), 'constant',
                          constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            new_img = pad(new_img, ((simplify(glx), simplify(grx)), (simplify(gly), simplify(gry))), 'constant',
                          constant_values=((0, 0), (0, 0)))
        return new_img

    def index_to_coords(self, index):
        x = index % self.x
        y = int((index / self.x) % self.y)
        return x, y, z

    def __getitem__(self, index):
        def transform_img(img):
            img = Image.fromarray(img)
            return self.transform(img)

        x = index % self.x
        y = int((index / self.x) % self.y)

        img256 = self.crop_with_padding(self.img, x, y, 256)

        return transform_img(img256), x, y

    def __len__(self):
        return self.len


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 9)
    model = torch.nn.DataParallel(model).cuda()


    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading model weights '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded model weights '{}' (epoch {})"
                  .format(args.weights, checkpoint['epoch']))
        else:
            print("=> no model weights found at '{}'".format(args.weights))
            return

    cudnn.benchmark = True

    image_folder = args.input_folder

    for i, image_path in enumerate(os.listdir(image_folder)):
        print('{} of {} images processed'.format(i, len(os.listdir(image_folder))))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_path = os.path.join(image_folder, image_path)
        image_ds = ImageLoader(image_path,
                               args.resize,
                               transforms.Compose([
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
        image_loader = torch.utils.data.DataLoader(image_ds,
                                                   batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True)

        pred_image = get_preds(image_loader, model, image_ds.x, image_ds.y)
        pred_jpg = np.zeros((image_ds.x, image_ds.y, 3), dtype=np.uint8)
        pred_jpg[:, :, 0] = pred_image.astype(np.uint8)
        pred_jpg = cv2.resize(pred_jpg, (image_ds.orig_y, image_ds.orig_x), interpolation=cv2.INTER_NEAREST)
        preds_jpg = remap_indices(pred_jpg)
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        save_file = os.path.join(args.output_folder, os.path.basename(image_path).split('.')[0]+'_output.png')
        cv2.imwrite(save_file, preds_jpg)
    print('{} of {} images processed'.format(i+1, len(os.listdir(image_folder))))


def remap_indices(gt_img):
    gt_img_proc = np.zeros((gt_img.shape[0], gt_img.shape[1], 3))
    # Find Background
    locs = np.where(gt_img[:, :, 0] == 0)
    gt_img_proc[locs[0], locs[1], 0] = 1

    # Find Text
    locs = np.where(gt_img[:, :, 0] == 1)
    gt_img_proc[locs[0], locs[1], 0] = 8


    # Find Decoration
    locs = np.where(gt_img[:, :, 0] == 2)
    gt_img_proc[locs[0], locs[1], 0] = 4


    # Find Comments
    locs = np.where(gt_img[:, :, 0] == 3)
    gt_img_proc[locs[0], locs[1], 0] = 2

    # Find Text+Decoration
    locs = np.where(gt_img[:, :, 0] == 4)
    gt_img_proc[locs[0], locs[1], 0] = 12

    # Find Text+Comments
    locs = np.where(gt_img[:, :, 0] == 5)
    gt_img_proc[locs[0], locs[1], 0] = 10


    # Find Text+Decoration
    locs = np.where(gt_img[:, :, 0] == 6)
    gt_img_proc[locs[0], locs[1]] = 12

    # Find Decoration+Comment
    locs = np.where(gt_img[:, :, 0] == 7)
    gt_img_proc[locs[0], locs[1]] = 6


    # Find Text+Decoration+Comment
    locs = np.where(gt_img[:, :, 0] >= 8)
    gt_img_proc[locs[0], locs[1]] = 14
    return gt_img_proc

def get_preds(image_loader, model, x, y):
    preds = np.zeros((x, y))
    # switch to evaluate mode
    model.eval()

    for i, (input, xb, yb) in enumerate(image_loader):

        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)
        for p, x, y in zip(output.data.cpu().numpy(), xb, yb):
            p = np.argmax(p)
            preds[x,y] = p

    return preds


if __name__ == '__main__':
    main()
