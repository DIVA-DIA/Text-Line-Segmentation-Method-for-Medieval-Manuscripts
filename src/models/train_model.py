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
import torchvision.models as models

import tensorboardX

import numpy as np
from PIL import Image
from skimage.util import pad

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('--log-dir',
                    help='where to save logs', default='./data/')
parser.add_argument('--log-folder',
                    help='override default log folder (to resume logging of experiment)',
                    default=None,
                    type=str)
parser.add_argument('--experiment-name',
                    help='provide a meaningful and descriptive name to this run',
                    default=None, type=str)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-train-patches', default=100000, type=int,
                    metavar='N', help='Number of patches to be used for training')
parser.add_argument('--num-val-patches', default=100000, type=int,
                    metavar='N', help='Number of patches to be used for validation')

best_prec1 = 0

args = parser.parse_args()
# Experiment name override
if args.experiment_name is None:
    vars(args)['experiment_name'] = input("Experiment name:")

# Set up tensorboard
basename = args.log_dir
experiment_name = args.experiment_name
if not args.log_folder:
    log_folder = os.path.join(basename, experiment_name, '{}'.format(time.strftime('%y-%m-%d-%Hh-%Mm-%Ss')))
else:
    log_folder = args.log_folder
logfile = 'logs.txt'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
writer = tensorboardX.SummaryWriter(log_dir=log_folder)


class DatasetInMem():
    def __init__(self, data_path, transform=None):
        self.data = np.load(data_path)
        self.transform = transform
        self.x = self.data[0][0].shape[0]
        self.y = self.data[0][0].shape[1]
        self.z = len(self.data)
        self.len = self.x * self.y * self.z
        self.class_counts, self.class_weights, self.class_target_weights = self.compute_class_weights()


    def generate_labels(self):
        labels = np.zeros(self.len, dtype=np.uint8)
        idx = 0
        for i in range(len(self.data)):
            labels[idx:idx + self.x * self.y] = self.data[i][1].flatten(order='F')
            idx += self.x * self.y
        return labels


    def compute_class_weights(self):
        tmp = [np.unique(item[1], return_counts=True) for item in self.data]
        counts = np.zeros((8))
        for index, count in tmp:
            counts[index] += count
        class_counts = counts
        counts = counts/np.sum(counts)
        target_weight = 1.0 / 8

        class_target_weights = [target_weight / item if item != 0 else 0 for item in counts]
        class_target_weights = class_target_weights/np.sum(class_target_weights)
        return class_counts, counts, class_target_weights

    def crop_with_padding(self, img, x, y, SIZE):
        def simplify(num):
            if num < 0:
                return abs(num)
            else:
                return 0

        SIZE = int(SIZE / 2)
        glx, gly, grx, gry = x - SIZE, y - SIZE, img.shape[0] - (x + SIZE), img.shape[1] - (y + SIZE)
        # print('{} {} {} {}'.format(simplify(glx), simplify(grx), simplify(gly), simplify(gry)))

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
        z = int(index / (self.x * self.y))
        return x, y, z

    def target_at_index(self, index):
        x, y, z = self.index_to_coords(index)
        target = self.data[z][1][x, y]
        return target

    def __getitem__(self, index):
        def transform_img(img):
            img = Image.fromarray(img)
            return self.transform(img)

        x = index % self.x
        y = int((index / self.x) % self.y)
        z = int(index / (self.x * self.y))
        target = self.data[z][1][x, y]



        img = self.data[z][0]
        img224 = self.crop_with_padding(img, x, y, 224)
        img32 = []
        return transform_img(img224), target

    def __len__(self):
        return self.len

def main():
    global args, best_prec1


    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    model.fc = nn.Linear(512, 8)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train.npy')
    valdir = os.path.join(args.data, 'public-test.npy')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_ds = DatasetInMem(valdir, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    # val_sampler = torch.utils.data.sampler.SubsetRandomSampler(np.random.randint(0, len(val_ds), args.num_val_patches))
    class_weights = val_ds.class_target_weights
    reciprocal_weights = [class_weights[item] for item in val_ds.generate_labels()]
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(reciprocal_weights, args.num_val_patches)
    del reciprocal_weights

    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True,
                                             sampler=val_sampler)


    if args.evaluate:
        validate(val_loader, model, criterion, args.start_epoch)
        return

    train_ds = DatasetInMem(traindir, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    class_weights = train_ds.class_target_weights
    reciprocal_weights = [class_weights[item] for item in train_ds.generate_labels()]
    #
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(reciprocal_weights, args.num_train_patches)
    del reciprocal_weights

    train_loader = torch.utils.data.DataLoader(train_ds
                                               ,
                                               batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True,
                                               sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=os.path.join(log_folder, 'checkpoint.pth.tar'))
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True).long()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # import pdb; pdb.set_trace()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        writer.add_scalar('train/mb_loss', loss.data[0], epoch * len(train_loader) + i)
        writer.add_scalar('train/mb_accuracy', prec1.cpu().numpy(), epoch * len(train_loader) + i)
    writer.add_scalar('train/accuracy', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True).long()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
        # Add loss and accuracy to Tensorboard
        writer.add_scalar('val/mb_loss', loss.data[0],
                          epoch * len(val_loader) + i)
        writer.add_scalar('val/mb_accuracy', prec1.cpu().numpy(),
                          epoch * len(val_loader) + i)

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    writer.add_scalar('val/accuracy', top1.avg, epoch - 1)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
