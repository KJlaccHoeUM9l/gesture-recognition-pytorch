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

import lstm.src.training.dataset as dataset
from lstm.src.training.lstm_arch import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith('__'))

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data', default='root/', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int)
parser.add_argument('--prefix', default='000000', type=str)
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--lstm_layers', default=1, type=int, metavar='LSTM',
                    help='number of lstm layers')
parser.add_argument('--hidden_size', default=512, type=int, metavar='HIDDEN',
                    help='output size of LSTM hidden layers')
parser.add_argument('--lr_step', default=10, type=float,
                    help='learning rate decay frequency')
parser.add_argument('--optim', '--optimizer', default='sgd',type=str,
                    help='optimizer: sgd | adam')
parser.add_argument('--fc_size', default=1024, type=int,
                    help='size of fully connected layer before LSTM')

best_prec1 = 0


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
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
        input_var, target_var = input_var.cuda(), target_var.cuda()

        # Run forward
        output, _ = model(input_var[0])
        target_var = target_var.repeat(output.shape[0])
        loss_t = criterion(output, target_var)
        weight = Variable(torch.Tensor(range(output.shape[0])) / (output.shape[0] - 1)).cuda()

        loss = torch.mean(loss_t * weight)  # Среднее между ошибкой на каждом кадре и его весом
        losses.update(loss.data, input.size(0))

        # Backprop and perform optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('\tTrain: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'lr {lr:.5f}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(train_loader), batch_time=batch_time,
                data_time=data_time, lr=optimizer.param_groups[-1]['lr'],
                loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, print_freq):
    #print('validate')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    fps = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # CUDA
        input_var, target_var = input_var.cuda(), target_var.cuda()

        # compute output
        output, _ = model(input_var[0])
        weight = Variable(torch.Tensor(range(output.shape[0])) / (output.shape[0] - 1)).cuda()
        output = torch.sum(output * weight.unsqueeze(1), dim=0, keepdim=True)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data[0].cpu(), target, topk=(1, min(model.num_classes, 5)))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        fps.update(float(input.size(1) / batch_time.val), input.size(0))

        if i % print_freq == 0:
            print('\tTest: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'fps {fps.val: .3f} ({fps.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), fps=fps, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))


        # Работало
        # output, _ = model(input_var[0])
        # _, predicted = torch.max(output.data, 1)
        # total += len(predicted)
        # correct += (predicted == int(target[0].data)).sum().item()
        #######################
        # target_var = target_var.repeat(output.shape[0])
        # loss_t = criterion(output, target_var)
        # weight = Variable(torch.Tensor(range(output.shape[0])) / (output.shape[0] - 1)).cuda()
        # loss = torch.mean(loss_t * weight)

    return top1.avg


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
            print('Error: file not found')


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #print('\taccuracy')

    maxk = max(topk)
    batch_size = target.size(0)

    if len(np.array(output.size())) == 1:
        _, pred = output.topk(2, -1, True, True)
        pred = pred.view(-1, 1)
    else:
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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

    train_dataset = dataset.CLMarshallingDataset(traindir, transform)
    valid_dataset = dataset.CLMarshallingDataset(valdir, transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # load and create model
    print("==> creating model '{}' ".format(args.arch))
    if args.pretrained:
        print("==> using pre-trained model '{}' ".format(args.arch))

    original_model = models.__dict__[args.arch](pretrained=args.pretrained) 
    model = FineTuneLstmModel(original_model, args.arch, 
        len(train_dataset.classes), args.lstm_layers, args.hidden_size, args.fc_size)
    
    #print(model)

    model.cuda()

    # loss criterion and optimizer
    criterion = nn.CrossEntropyLoss(reduce=False)
    criterion = criterion.cuda()

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
        print('Epoch #' + (epoch + 1).__str__())
        optimizer = adjust_learning_rate(optimizer, epoch)

        # train on one epoch
        loss1 = train(train_loader, model, criterion, optimizer, epoch, args.print_frec)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, args.print_frec)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'lstm_layers': args.lstm_layers,
            'hidden_size': args.hidden_size,
            'fc_size': args.fc_size,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict()}, is_best, '../../results/weights/', args.prefix, str(epoch + 1))

        loss_list.append(loss1)
        acc_list.append(prec1)
        currentTime = time.time() - start
        avgTime.update(currentTime)
        print('\tTime left: ' + str(round(avgTime.avg * (numEpoch - epoch - 1) / 60, 1)) + ' minutes')
        if epoch == 0:
            avgTime.reset()


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


if __name__ == '__main__':
    print(torch.cuda.get_device_name(0))

    args = parser.parse_args()
    args.data = 'C:/neural-networks/datasets/TestUAVGesture/frames-short-70-cut-224-part/'
    args.prefix = getPrefix()
    args.arch = 'alexnet'
    args.batch_size = 1
    args.lr = 0.1
    args.lr_step = 7
    args.epochs = 7
    args.optim = 'sgd'
    args.print_frec = 10
    print(args)

    totalStart = time.time()
    main(args)
    print('Total time: ' + str(round((time.time() - totalStart) / 60)) + ' minutes')
