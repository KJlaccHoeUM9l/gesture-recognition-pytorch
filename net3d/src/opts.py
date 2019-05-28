import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='C:\\neural-networks\\datasets\\UAV_activity_net\\',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        default='C:\\neural-networks\\datasets\\UAV_activity_net\\UAVGesture\\jpg\\',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_directory',
        default='annotation',
        type=str,
        help='Annotation directory path')
    parser.add_argument(
        '--annotation_path',
        default='UAVGesture_2.json',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--no_cross_validation',
        default=False,
        type=bool,
        help='If true, no cross validation will be performed')
    parser.add_argument(
            '--frequence_cross_validation',
            default=1,
            type=int,
            help='Frequence of switching between data loaders')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='UAV',
        type=str,
        help='Used dataset (UAV)')
    parser.add_argument(
        '--n_classes',
        default=13,
        type=int,
        help=
        'Number of classes (UAVGesture: 13)'
    )
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales',
        default=5,
        type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.84089641525,  # 2^(-1/4)
        type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='corner',
        type=str,
        help=
        'Spatial cropping method in training. Corner is selection from 4 corners and 1 center. (corner | center)'
    )
    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
            '--frequence_regulate_lr',
            default=15,
            type=int,
            help='Regulate lr every (value) epoch')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument(
        '--mean_dataset',
        default='uav_gesture',
        type=str,
        help='dataset for mean values of mean subtraction (uav_gesture)')
    parser.add_argument(
        '--std_dataset',
        default='uav_gesture',
        type=str,
        help='dataset for std values (uav_gesture)')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=50,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=20,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext)')
    parser.add_argument(
        '--model_depth',
        default=34,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')

    args = parser.parse_args()

    return args
