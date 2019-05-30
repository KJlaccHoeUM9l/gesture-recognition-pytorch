import os
import random


class Object(object):
    pass


def valid_videos(count_videos, top_boundary):
    nums = set()

    while len(nums) != count_videos:
        nums.add(random.randint(1, top_boundary))

    return nums


def rest_validation(available_videos, size):
    nums = set()

    min_num = min(available_videos)
    max_num = max(available_videos)
    while len(nums) != size:
        try_i = random.randint(min_num, max_num)
        if available_videos.__contains__(try_i):
            nums.add(try_i)

    return nums


def get_split(video_list, validation_set):
    split = []

    for video in video_list:
        words = video.split('_')
        index = words[len(words) - 1]

        if validation_set.__contains__(int(index)) is True:
            split.append(video + ' 2')      # Validation
        else:
            split.append(video + ' 1')      # Training

    return split


def save_split(save_name, split):
    with open(save_name, "w") as file:
        for line in split:
            file.write(line + '\n')


def main(opt):
    folders = os.listdir(opt.video_root_directory_path)

    for video_folder in folders:
        validation_list = []
        split_list = []

        video_list = os.listdir(os.path.join(opt.video_root_directory_path, video_folder))
        valid_videos_count = round(len(video_list) / opt.split_quantities - 0.5)
        available_videos_in_folder = set(range(1, len(video_list) + 1))

        # Start separation
        first_validation_set = valid_videos(valid_videos_count, len(video_list))
        validation_list.append(first_validation_set)
        for i in range(1, opt.split_quantities):
            available_videos_in_folder -= validation_list[i - 1]
            validation_list.append(rest_validation(available_videos_in_folder, valid_videos_count))

        for validation_set in validation_list:
            split_list.append(get_split(video_list, validation_set))

        num_split = 1
        for split in split_list:
            save_name = os.path.join(opt.save_root_directory_path,
                                     '{}_test_split_{}.txt'.format(video_folder, num_split))
            save_split(save_name, split)
            num_split += 1


if __name__ == '__main__':
    opt = Object()
    opt.video_root_directory_path = 'C:\\neural-networks\\datasets\\UAV_activity_net\\UAVGesture\\jpg\\'
    opt.save_root_directory_path = 'C:\\neural-networks\\datasets\\TestUAVGesture\\annotation_test\\'
    opt.split_quantities = 3

    print('Start separate:')
    main(opt)
    print('Separate ended success!')
