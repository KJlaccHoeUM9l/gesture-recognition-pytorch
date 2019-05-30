import os
from PIL import Image
import numpy as np
import time


def main(video_root_directory_path):
    folders = os.listdir(video_root_directory_path)

    video_limit = 2
    frame_limit = 25
    row_limit = 224

    values_r_channel = []
    values_g_channel = []
    values_b_channel = []

    for videoFolder in folders:
        print('Folder: ' + videoFolder)

        video_list = os.listdir(video_root_directory_path + videoFolder + '/')
        num_video = 0
        for video in video_list:
            if num_video == video_limit:
                break
            num_video += 1
            print('\tVideo: ' + video)

            image_list = os.listdir(video_root_directory_path + videoFolder + '/' + video + '/')
            image_list.remove('n_frames')

            num_frame = 0
            for image in image_list:
                if num_frame == frame_limit:
                    break
                num_frame += 1
                image_path = video_root_directory_path + videoFolder + '/' + video + '/' + image
                with open(image_path, 'rb') as f:
                    with Image.open(f) as img:
                        array_image = np.array(img)
                        num_row = 0
                        for row in array_image:
                            if num_row == row_limit:
                                break
                            num_row += 1
                            for R, G, B in row:
                                values_r_channel.append(R)
                                values_g_channel.append(G)
                                values_b_channel.append(B)

    print('\n\n\nStart calculating of mean...')
    mean_r = np.mean(values_r_channel)
    mean_g = np.mean(values_g_channel)
    mean_b = np.mean(values_b_channel)
    print('\tMean R: ' + str(mean_r))
    print('\tMean G: ' + str(mean_g))
    print('\tMean B: ' + str(mean_b))

    print('Start calculating of std...')
    std_r = np.std(values_r_channel)
    std_g = np.std(values_g_channel)
    std_b = np.std(values_b_channel)
    print('\tStd R: ' + str(std_r))
    print('\tStd G: ' + str(std_g))
    print('\tStd B: ' + str(std_b))


if __name__ == '__main__':
    video_root_directory_path = 'C:\\neural-networks\\datasets\\TestUAVGesture\\test\\'

    print('Start...')
    totalStart = time.time()
    main(video_root_directory_path)
    print('Total time: ' + str(round((time.time() - totalStart) / 60)) + ' minutes')
    print('Ended success!')
