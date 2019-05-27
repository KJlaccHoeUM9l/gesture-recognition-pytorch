import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
     Compose, Normalize, Scale, CenterCrop, MultiScaleCornerCrop,
     RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel
from dataset import get_training_set, get_validation_set
from utils import Logger
from train import train_epoch
from validation import val_epoch

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value, dataset=opt.std_dataset)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)

    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if not opt.no_mean_norm:
        norm_method = Normalize(opt.mean, opt.std)
    else:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])

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
                                 ToTensor(opt.norm_value), norm_method])
    # Временное преобразование
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    # Целевое преобразование
    target_transform = ClassLabel()

    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads,
                                               pin_memory=True)
    train_logger = Logger(os.path.join(opt.result_path, 'train.log'),
                          ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(os.path.join(opt.result_path, 'train_batch.log'),
                                ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    optimizer = optim.SGD(parameters,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          dampening=opt.dampening,
                          weight_decay=opt.weight_decay)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

    # ***************************** VALIDATION CONFIGURATIONS *********************************
    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(opt.norm_value), norm_method])
    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(validation_data,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True)
    val_logger = Logger(os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    # **************************************** TRAINING ****************************************
    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
