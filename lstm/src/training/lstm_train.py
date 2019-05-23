import argparse
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F

import lstm.src.training.lstm_dataset as dataset
from lstm.src.training.lstm_arch import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Training')
# data parameters
parser.add_argument('--data', default='root/', type=str)

# model parameters
parser.add_argument('--arch', metavar='ARCH', default='alexnet', help='model architecture' + ' (default: alexnet)')
parser.add_argument('--lstm_layers', default=1, type=int, metavar='LSTM', help='number of lstm layers')
parser.add_argument('--hidden_size', default=512, type=int, metavar='HIDDEN', help='output size of LSTM hidden layers')
parser.add_argument('--fc_size', default=1024, type=int, help='size of fully connected layer before LSTM')

# train parameters
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay(default: 1e-4)')
parser.add_argument('--lr_step', default=10, type=float, help='learning rate decay frequency')
parser.add_argument('--optim', '--optimizer', default='sgd',type=str, help='optimizer: sgd | adam')

# other parameters
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--print-freq', '-p', default=50, type=int)
parser.add_argument('--prefix', default='000000', type=str)

best_prec1 = 0


def train(train_loader, model, criterion, optimizer, print_freq):
    #print('train')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # wrap inputs and targets in Variable
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # CUDA
        input_var, target_var = input_var.to(device), target_var.to(device)

        # Run forward
        output, _ = model(input_var[0])
        target_var = target_var.repeat(output.shape[0])
        loss_t = criterion(output, target_var)
        weight = Variable(torch.Tensor(range(output.shape[0])) / (output.shape[0])).to(device)

        loss = torch.mean(loss_t * weight)  # Среднее между ошибкой на каждом кадре и его весом
        losses.update(loss.data, input.size(0))

        # output, _ = model(input_var[0])
        # weight = Variable(torch.Tensor(range(output.shape[0])) /
        #                   (output.shape[0])).cuda().view(-1, 1).repeat(1, output.shape[1])
        # output = torch.mul(output, weight)
        # output = torch.mean(output, dim=0).unsqueeze(0)  # Среднее между ошибкой на каждом кадре и его весом
        #
        # loss = criterion(output, target_var)
        # losses.update(loss.item(), input.size(0))

        # Backprop and perform optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print_log('\tTrain: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'lr {lr:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, lr=optimizer.param_groups[-1]['lr'],
                      loss=losses))

    return losses.avg


def validate(val_loader, model, print_freq):
    #print('validate')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    correct = 0
    total = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # CUDA
        input_var, target_var = input_var.to(device), target_var.to(device)

        # compute output
        output, _ = model(input_var[0])
        weight = Variable(torch.Tensor(range(output.shape[0])) / (output.shape[0])).to(device)
        output = torch.sum(output * weight.unsqueeze(1), dim=0)
        output = nn.functional.softmax(output, dim=0)

        _, predicted = torch.max(output.data.cpu(), 0)
        if int(predicted) == int(target[0].data):
            correct += 1
        total += 1

        if i % print_freq == 0:
            print_log('\tTest: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Accuracy {acc:.3f}'.format(
                        i, len(val_loader),  batch_time=batch_time, acc=(correct/total)*100))

    return (correct / total) * 100


def DelFiles(path, code):
    deleteList = []
    files = os.listdir(path)

    for file in files:
        if file.find(code) >= 0:
            deleteList.append(file)

    for record in deleteList:
        if os.path.isfile(path + record):
            os.remove(path + record)
        else:
            print_log('Error: file not found')


def save_checkpoint(state, is_best, path, prefix, epoch):
    #print('\tsave_checkpoint')
    DelFiles(path, prefix + '_checkpoint')
    filename = path + prefix + '_checkpoint_' + epoch + '_epoch.pth.tar'
    torch.save(state, filename)

    if is_best:
        DelFiles(path, prefix + '_model_best')
        shutil.copyfile(filename, path + prefix + '_model_best_' + epoch + '_epoch.pth.tar')


class AverageMeter(object):
    '''computes and stores the average and current value'''
    def __init__(self):
        #print('\tAverageMeter.__init__()')
        self.reset()

    def reset(self):
        #print('\tAverageMeter.reset()')
        self.val = 0
        self.avg = 0
        self.max = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        #print('\tAverageMeter.update')
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #print('\tadjust_learning_rate')
    if not epoch % args.lr_step and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def main(args):
    global best_prec1

    # Data Transform and data loading
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.339, 0.224, 0.225])

    transform = (transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                    ]),
                transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()]
                                    )
                )

    train_dataset = dataset.UAVGestureDataset(traindir, transform)
    valid_dataset = dataset.UAVGestureDataset(valdir, transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # load and create model
    print_log("==> creating model '{}' ".format(args.arch))
    original_model = models.__dict__[args.arch](pretrained=False)#True)
    model = CNN_LSTM_Model(original_model, args.arch, len(train_dataset.classes),
                           args.lstm_layers, args.hidden_size, args.fc_size)
    model.to(device)

    # loss criterion and optimizer
    criterion = nn.CrossEntropyLoss(reduce=False)
    criterion = criterion.to(device)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.features.parameters(), 'lr': 0.1 * args.lr}, 
                                    {'params': model.fc_pre.parameters()},
                                    {'params': model.rnn.parameters()}, {'params': model.fc.parameters()}],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam([{'params': model.features.parameters(), 'lr': 0.1 * args.lr}, 
                                    {'params': model.fc_pre.parameters()}, 
                                    {'params': model.rnn.parameters()}, {'params': model.fc.parameters()}],
                                    lr=args.lr, weight_decay=args.weight_decay)

    # Training on epochs
    loss_list = []
    acc_list = []
    numEpoch = args.epochs

    avgTime = AverageMeter()
    for epoch in range(0, numEpoch):
        start = time.time()
        print_log('Epoch #' + (epoch + 1).__str__())
        optimizer = adjust_learning_rate(optimizer, epoch)

        # train on one epoch
        loss = train(train_loader, model, criterion, optimizer, args.print_freq)

        # evaluate on validation set
        prec = validate(val_loader, model, args.print_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec > best_prec1
        best_prec1 = max(prec, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'lstm_layers': args.lstm_layers,
            'hidden_size': args.hidden_size,
            'fc_size': args.fc_size,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict()}, is_best, '../../results/weights/', args.prefix, str(epoch + 1))

        loss_list.append(loss)
        acc_list.append(prec)
        currentTime = time.time() - start
        avgTime.update(currentTime)
        print_log('\tTime left: ' + str(round(avgTime.avg * (numEpoch - epoch - 1) / 60, 1)) + ' minutes')
        if epoch == 0:
            avgTime.reset()

    print_log('Min loss: ' + str(float(min(loss_list))))
    print_log('Max loss: ' + str(float(max(loss_list))))
    print_log('Min accuracy: : ' + str(min(acc_list)))
    print_log('Max accuracy: : ' + str(max(acc_list)))

    # Saving results of training
    SavePictures(np.linspace(1, numEpoch, numEpoch), loss_list, 'red', 'Loss',
                 '../../results/images/' + args.prefix + '_' + model.modelName + '_'
                 + str(numEpoch) + '_epochs' + '_loss' + '.png')
    SavePictures(np.linspace(1, numEpoch, numEpoch), acc_list, 'blue', 'Accuracy',
                 '../../results/images/' + args.prefix + '_' + model.modelName + '_'
                 + str(numEpoch) + '_epochs' + '_accuracy' + '.png')


def getPrefix():
    f = open('NumberOfStart.txt', 'r');
    numStart = int(f.read());
    f.close()
    f = open('NumberOfStart.txt', 'w');
    f.write(str(numStart + 1));
    f.close()

    return str(numStart).zfill(4)


def SavePictures(axis_x, axis_y, lineColor, lineLabel, name):
    fig, ax1 = plt.subplots()
    ax1.plot(axis_x, axis_y, color=lineColor, label=lineLabel)
    ax1.set_xlabel("Epoch")
    ax1.legend()
    fig.savefig(name)


logfile = None
logpath = 'log.txt'
def print_log(*args):
    global logfile
    print(*args)
    if logfile is None:
        logfile = open(logpath, 'a')
    print(*args, file=logfile)


if __name__ == '__main__':
    #print(torch.cuda.get_device_name(0))

    args = parser.parse_args()
    args.data = 'C:/neural-networks/datasets/TestUAVGesture/frames-short-70-cut-224-part/'
    args.prefix = getPrefix()
    #args.arch = 'alexnet'
    #args.arch = 'resnet18'
    #args.arch = 'resnet50'
    args.batch_size = 1
    args.lr = 0.1
    args.lr_step = 7
    args.epochs = 2
    args.optim = 'sgd'
    args.print_freq = 10

    logpath = '../../results/logs/' + args.prefix + '_' + args.arch + '_' + str(args.epochs) + '_epochs_' + args.optim + '.txt'
    print_log('Device: ' + str(device))
    print_log(args)

    totalStart = time.time()
    main(args)
    print_log('Total time: ' + str(round((time.time() - totalStart) / 60)) + ' minutes')
