import cv2
import os
import time

import net3d.src.utils.body_detection as bd


class Object(object):
    pass


def extract_images(opt, video_path, save_path):
    k = 2
    new_w = int(1920 / k)
    new_h = int(1080 / k)

    if not opt.no_body_detection:
        x, y, w, h = bd.get_rectangle(video_path, opt.frame_limit, new_w, new_h)
        x, y, w, h = bd.get_square(x, y, w, h)

    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    frame_count = 0
    while success and frame_count < opt.frame_limit:
        image = cv2.resize(image, (new_w, new_h))
        if not opt.no_body_detection:
            image = bd.cut_image(image, x, y, w, h, opt.final_size)

        frame_save_path = os.path.join(save_path, 'img{}.jpg'.format(str(frame_count).zfill(6)))
        cv2.imwrite(frame_save_path, image)

        success, image = video_cap.read()
        frame_count += 1


def main(opt):
    num_current_class = 0
    class_avg_time = 0

    class_folders = os.listdir(opt.video_root_directory_path)
    for class_folder in class_folders:
        class_name = '_'.join(class_folder.lower().split(' '))
        class_save_path = os.path.join(opt.save_root_directory_path, class_name)
        if not os.path.exists(class_save_path):
            os.makedirs(class_save_path)

        current_class_video_path = os.path.join(opt.video_root_directory_path, class_folder)
        current_video_list = os.listdir(current_class_video_path)

        num_video = 0
        class_start = time.time()
        for video in current_video_list:
            video_source_path = os.path.join(current_class_video_path, video)
            video_save_path = os.path.join(class_save_path, '{}_video_{}'.format(class_name, num_video + 1))
            if not os.path.exists(video_save_path):
                os.makedirs(video_save_path)

            # Раскадровка
            extract_images(opt, video_source_path, video_save_path)

            num_video += 1
            if num_video == opt.video_limit:
                break

        class_avg_time = (class_avg_time + (time.time() - class_start)) / 2
        num_current_class += 1
        print('*************************')
        print('Done:')
        print('\tFolders: ' + str(num_current_class) + '/' + str(opt.num_classes))
        print('\tTime left: ' + str(round(class_avg_time * (opt.num_classes - num_current_class) / 60, 1)) + ' minutes')

    print('*************************')


if __name__ == '__main__':
    opt = Object()
    opt.video_root_directory_path = 'C:\\neural-networks\\datasets\\UAVGesture\\'
    opt.save_root_directory_path = 'C:\\neural-networks\\datasets\\TestUAVGesture\\test\\'
    opt.num_classes = 13
    opt.no_body_detection = False
    opt.final_size = 224
    opt.frame_limit = 75
    opt.video_limit = 7

    print('Storyboard started...')
    total_start = time.time()
    main(opt)
    print('Total time: ' + str(round((time.time() - total_start) / 60)) + ' minutes')
    print('Storyboard ended success!')
