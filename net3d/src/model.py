import torch
from torch import nn

from net3d.src.models import resnet, pre_act_resnet, wide_resnet, resnext


def generate_model(opt):
    assert opt.model in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'c3d']

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    elif opt.model == 'resnext':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50]

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model, model.parameters()
