import argparse
import os
import shutil
import logging
import time
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import MySTCNN

import numpy as np
import matplotlib.pyplot as plt

from utils import AverageMeter, accuracy, Bar

parser = argparse.ArgumentParser()

# datasets
parser.add_argument('-d', '--dataset', default='casme2', type=str)
# parser.add_argument('--dataset-path', default='dataset/CASME2_224_15frames.pickle')
parser.add_argument('--dataset-path', default='dataset/CASME2_BGR_224_15frames.pickle')
parser.add_argument('-f', '--folds', default=10, type=int, help='k-folds cross validation')

# optimization options
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=2, type=int, metavar='N',
                help='train batchsize')
parser.add_argument('--test-batch', default=2, type=int, metavar='N',
                help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 30, 40],
                help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.90, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                metavar='W', help='weight decay (default: 1e-4)')

# checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints/casme2_mystcnn', type=str, metavar='PATH',
                help='path to save checkpoint (default:checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH')

# architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='')

# miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch

    # 创建 checkpoint 目录
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    # load data
    print('==> Preparing dataset %s' % args.dataset)
    with open(args.dataset_path, 'rb') as f:
        data = pickle.load(f)
    # model
    print("==> creating model '{}'".format(args.arch))
    model = MySTCNN()
    if use_cuda:
        model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # set up logging
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.checkpoint, 'log_info.log'),
                        filemode='a+',
                        format="%(asctime)-15s %(levelname)-8s  %(message)s")
    
    # log configuration
    logging.info('-' * 10 + 'configuration' + '*' * 10)
    for arg in vars(args):
        logging.info((arg, str(getattr(args, arg))))

    # 10-fold cv
    acc_fold = []
    reset_lr = state['lr']
    for f_num in range(args.folds):
        state['lr'] = reset_lr
        model = MySTCNN()
        if use_cuda:
            model = model.cuda()
        model.reset_all_weights()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        average_acc = 0
        best_acc = 0

        # prepare input
        train_img, train_label, test_img, test_label = data[f_num]['train_img'], data[f_num]['train_label'], data[f_num]['test_img'], data[f_num]['test_label']

        train_img = torch.tensor(train_img, dtype=torch.float) / 255.0  # (b_s, frames, h, w)
        train_img = train_img.permute(0, 4, 1, 2, 3)
        # train_img = train_img.unsqueeze(1)

        test_img = torch.tensor(test_img, dtype=torch.float) / 255.0
        test_img = test_img.permute(0, 4, 1, 2, 3)
        # test_img = test_img.unsqueeze(1)

        train_label, test_label = torch.tensor(train_label, dtype=torch.long), torch.tensor(test_label, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(train_img, train_label)
        train_iter = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch,
            shuffle=True
        )

        test_dataset = torch.utils.data.TensorDataset(test_img, test_label)
        test_iter = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch,
            shuffle=False
        )
        # train and val
        for epoch in range(start_epoch, args.epochs):
            # 在特定的epoch 调整学习率
            adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
            
            train_loss, train_acc = train(train_iter, model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test(test_iter, model, criterion, epoch, use_cuda)

            # logger

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, f_num, checkpoint=args.checkpoint)
        
        # compute average acc
        acc_fold.append(best_acc)
        average_acc = sum(acc_fold) / len(acc_fold)

        logging.info('fold: %d, best_acc: %.2f, average_acc: %.2f' % (f_num, best_acc, average_acc))
    


def train(train_iter, model, criterion, optimizer, epoch, user_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_iter))
    for batch_idx, (inputs, targets) in enumerate(train_iter):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # compute output
        per_outputs = model(inputs)

        per_loss = criterion(per_outputs, targets)

        loss = per_loss

        # measure accuracy and record loss
        prec = accuracy(per_outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec[0].item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

         # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(inputs),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(test_iter, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(test_iter))
    for batch_idx, (inputs, targets) in enumerate(test_iter):
    # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        """
        np_inputs = inputs.numpy()
        np_att = attention.numpy()
        for item_in, item_att in zip(np_inputs, np_att):
            print(item_in.shape, item_att.shape)
        """

        # measure accuracy and record loss
        prec = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec[0].item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(inputs),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, f_num, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, 'fold_' + str(f_num) + '_' + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'fold_' + str(f_num) + '_model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma


if __name__ == '__main__':
    main()