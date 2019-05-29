import os
import time
import numpy as np
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from net3d.src.opts import parse_opts
from net3d.src.model import generate_model
from net3d.src.mean import get_mean, get_std
from net3d.src.spatial_transforms import (
     Compose, Normalize, Scale, CenterCrop, MultiScaleCornerCrop,
     RandomHorizontalFlip, ToTensor)
from net3d.src.temporal_transforms import LoopPadding, TemporalRandomCrop
from net3d.src.target_transforms import ClassLabel
from net3d.src.dataset import get_training_set, get_validation_set
from net3d.src.utils import Logger
from net3d.src.train import train_epoch
from net3d.src.validation import val_epoch
from net3d.src.utils import save_pictures, get_prefix, AverageMeter, regulate_learning_rate


def main():
    opt = parse_opts()
    # Path configurations
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_directory = os.path.join(opt.root_path, opt.annotation_directory)
        opt.annotation_path = os.path.join(opt.annotation_directory, opt.annotation_path)

        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        dir_name = os.path.join(opt.result_path,
                                get_prefix() + '_{}{}_{}_epochs'.format(opt.model, opt.model_depth, opt.n_epochs))
        os.mkdir(dir_name)
        opt.result_path = os.path.join(opt.result_path, dir_name)

    # For data generator
    opt.scales = [opt.initial_scale]
    for epoch in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    # Model
    model, parameters = generate_model(opt)
    # print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    # Normalizing
    if not opt.no_mean_norm:
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value, dataset=opt.std_dataset)
        norm_method = Normalize(opt.mean, opt.std)
    else:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])

    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # **************************** TRAINING CONFIGURATIONS ************************************
    assert opt.train_crop in ['corner', 'center']
    if opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])

    # Пространственное преобразование
    spatial_transform = Compose([crop_method,
                                 RandomHorizontalFlip(),
                                 ToTensor(opt.norm_value),
                                 norm_method])
    # Временное преобразование
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    # Целевое преобразование
    target_transform = ClassLabel()

    train_loader_list = []
    if not opt.no_cross_validation:
        annotation_list = os.listdir(opt.annotation_directory)
        for annotation in annotation_list:
            opt.annotation_path = os.path.join(opt.annotation_directory, annotation)
            training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
            train_loader = torch.utils.data.DataLoader(training_data,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=opt.n_threads,
                                                       pin_memory=True)
            train_loader_list.append(train_loader)
    else:
        training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(training_data,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.n_threads,
                                                   pin_memory=True)
        train_loader_list.append(train_loader)

    train_logger = Logger(os.path.join(opt.result_path, 'train.log'),
                          ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(os.path.join(opt.result_path, 'train_batch.log'),
                                ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    optimizer = optim.SGD(parameters,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          dampening=opt.dampening,
                          weight_decay=opt.weight_decay)

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

    # ***************************** VALIDATION CONFIGURATIONS *********************************
    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(opt.norm_value),
                                 norm_method])
    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = ClassLabel()

    val_loader_list = []
    if not opt.no_cross_validation:
        annotation_list = os.listdir(opt.annotation_directory)
        for annotation in annotation_list:
            opt.annotation_path = os.path.join(opt.annotation_directory, annotation)
            validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
            val_loader = torch.utils.data.DataLoader(validation_data,
                                                     batch_size=opt.batch_size,
                                                     shuffle=False,
                                                     num_workers=opt.n_threads,
                                                     pin_memory=True)
            val_loader_list.append(val_loader)
    else:
        validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(validation_data,
                                                 batch_size=opt.batch_size,
                                                 shuffle=False,
                                                 num_workers=opt.n_threads,
                                                 pin_memory=True)
        val_loader_list.append(val_loader)

    val_logger = Logger(os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    # **************************************** TRAINING ****************************************
    epoch_avg_time = AverageMeter()
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    current_train_data = 0
    current_valid_data = 0
    opt.frequence_cross_validation = round(opt.n_epochs / 3 + 0.5)
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        epoch_start_time = time.time()
        print('Epoch #' + str(epoch))

        optimizer = regulate_learning_rate(optimizer, epoch, opt.frequence_regulate_lr)

        train_loader = train_loader_list[current_train_data]
        if not opt.no_cross_validation and epoch % opt.frequence_cross_validation == 0:
            print('\t##### Cross-validation: switch training data #####')
            current_train_data = (current_train_data + 1) % len(train_loader_list)
            train_loader = train_loader_list[current_train_data]
        train_loss, train_acc = train_epoch(epoch, train_loader, model, criterion, optimizer, opt,
                                            train_logger, train_batch_logger)

        val_loader = val_loader_list[current_valid_data]
        if not opt.no_cross_validation and epoch % opt.frequence_cross_validation == 0:
            print('\t##### Cross-validation: switch validation data #####')
            current_valid_data = (current_valid_data + 1) % len(val_loader_list)
            val_loader = val_loader_list[current_valid_data]
        validation_acc = val_epoch(epoch, val_loader, model, criterion, opt, val_logger)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(validation_acc)

        epoch_end_time = time.time() - epoch_start_time
        epoch_avg_time.update(epoch_end_time)
        print('\tTime left: ' + str(round(epoch_avg_time.avg * (opt.n_epochs - epoch) / 60, 1)) + ' minutes')

    # ******************************* SAVING RESULTS OF TRAINING ******************************
    save_pictures(np.linspace(1, opt.n_epochs, opt.n_epochs), train_loss_list, 'red', 'Loss',
                  os.path.join(opt.result_path, 'train_loss.png'))
    save_pictures(np.linspace(1, opt.n_epochs, opt.n_epochs), train_acc_list, 'blue', 'Accuracy',
                  os.path.join(opt.result_path, 'train_accuracy.png'))
    save_pictures(np.linspace(1, opt.n_epochs, opt.n_epochs), valid_acc_list, 'blue', 'Accuracy',
                  os.path.join(opt.result_path, 'validation_accuracy.png'))


if __name__ == '__main__':
    total_start = time.time()
    main()
    print('Total time: ' + str(round((time.time() - total_start) / 60)) + ' minutes')
