def get_mean(norm_value=255, dataset='uav_gesture'):
    assert dataset in ['uav_gesture']

    if dataset == 'uav_gesture':
        return [139.1787 / norm_value, 134.1084 / norm_value, 120.7194 / norm_value]


def get_std(norm_value=255, dataset='uav_gesture'):
    assert dataset in ['uav_gesture']

    if dataset == 'uav_gesture':
        return [21.8274 / norm_value, 25.5777 / norm_value, 16.8124 / norm_value]
