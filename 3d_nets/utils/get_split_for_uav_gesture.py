import os
import random


def validVideos(countVideos, topBoundary):
    nums = set()

    while nums.__len__() != countVideos:
        nums.add(random.randint(1, topBoundary))

    return nums


def restValidation(availableVideos, size):
    nums = set()

    minNum = min(availableVideos)
    maxNum = max(availableVideos)
    while len(nums) != size:
        try_i = random.randint(minNum, maxNum)
        if availableVideos.__contains__(try_i):
            nums.add(try_i)

    return nums


def getSplit(video_list, validation_set):
    split = []

    for video in video_list:
        #name, index = video.split('_')
        words = video.split('_')
        index = words[len(words) - 1]

        if validation_set.__contains__(int(index)) is True:
            split.append(video + ' 2')
        else:
            split.append(video + ' 1')

    return split


def saveSplit(save_name, split):
    with open(save_name, "w") as file:
        for line in split:
            file.write(line + '\n')


def main(video_root_directory_path, save_root_directory_path):
    folders = os.listdir(video_root_directory_path)

    for videoFolder in folders:
        videoList = os.listdir(video_root_directory_path + videoFolder + '/')

        # Разделение данных на обучение и валидацию
        trainVideosCount = round(0.7 * videoList.__len__() + 0.5)  # 70%
        validVideosCount = videoList.__len__() - trainVideosCount

        all_videos_in_folder = set(range(1, len(videoList) + 1))

        validation1 = validVideos(validVideosCount, len(videoList))
        validation2 = restValidation(all_videos_in_folder - validation1, validVideosCount)
        validation3 = restValidation(all_videos_in_folder - validation1 - validation2, validVideosCount)

        split1 = getSplit(videoList, validation1)
        split2 = getSplit(videoList, validation2)
        split3 = getSplit(videoList, validation3)

        label = videoFolder.lower().split(' ')
        label = '_'.join(label)
        save_name_1 = save_root_directory_path + label + '_test_split1.txt'
        save_name_2 = save_root_directory_path + label + '_test_split2.txt'
        save_name_3 = save_root_directory_path + label + '_test_split3.txt'

        saveSplit(save_name_1, split1)
        saveSplit(save_name_2, split2)
        saveSplit(save_name_3, split3)


if __name__ == '__main__':
    video_root_directory_path = 'C:/neural-networks/datasets/UAV_activity_net/jpg/frames-short-70-cut-224-full/'
    save_root_directory_path = 'C:/neural-networks/datasets/UAV_activity_net/annotation/'

    print('Start separate:')
    main(video_root_directory_path, save_root_directory_path)
    print('Separate ended success!')
