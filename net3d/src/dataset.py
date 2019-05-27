from net3d.src.datasets.uav_gesture import UAVGesture


def get_training_set(opt, spatial_transform, temporal_transform, target_transform):

    assert opt.dataset in ['UAV']

    if opt.dataset == 'UAV':
        training_data = UAVGesture(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):

    assert opt.dataset in ['UAV']

    if opt.dataset == 'UAV':
        validation_data = UAVGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    return validation_data
