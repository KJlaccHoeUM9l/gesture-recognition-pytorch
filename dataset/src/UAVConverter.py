import cv2
import os
import time
import random

import src.BodyDetection as bd


def extractImages(video_path, save_path):
    finalSize = 224
    frameLimit = 70

    flagResize = True
    k = 2
    new_w = int(1920 / k)
    new_h = int(1080 / k)

    x, y, w, h = bd.getRectangle(video_path, frameLimit, flagResize, new_w, new_h)  ##
    x, y, w, h = bd.getSquare(x, y, w, h)                                           ##

    videoCap = cv2.VideoCapture(video_path)
    success, image = videoCap.read()
    frameCount = 0
    while success and frameCount < frameLimit:
        if flagResize:
            image = cv2.resize(image, (new_w, new_h))
        image = bd.cut_image(image, x, y, w, h, finalSize)                          ##
        #cv2.imshow('feed', image)
        cv2.imwrite(save_path + 'img' + str(frameCount).zfill(6) + '.jpg', image)  # save frame as JPEG file
        success, image = videoCap.read()
        frameCount += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # fps = 25
        # if (frameCount % fps == 0):
        #     print(frameCount / fps, ' second')


def numberOfVideos(root_directory_path):
    totalFolders = 0
    totalVideos = 0

    folders = os.listdir(root_directory_path)
    totalFolders += folders.__len__()

    for videoFolder in folders:
        videoList = os.listdir(root_directory_path + videoFolder + '/')
        totalVideos += videoList.__len__()

    return totalFolders, totalVideos


def validVideos(countVideos, topBoundary):
    nums = set()

    while nums.__len__() != countVideos:
        nums.add(random.randint(1, topBoundary))

    return nums


def main(video_root_directory_path, save_root_directory_path):
    # Для отображения прогресса нарезки
    print('Storyboard started...')
    totalFolders, totalVideos = numberOfVideos(video_root_directory_path)
    currentFolder = 0
    currentVideo = 0
    avgTime = 0

    # Начало нарезки
    folders = os.listdir(video_root_directory_path)
    for videoFolder in folders:
        video_path = video_root_directory_path + videoFolder + '/'

        videoList = os.listdir(video_root_directory_path + videoFolder + '/')

        # Разделение данных на обучение и валидацию
        trainVideosCount = round(0.8 * videoList.__len__() + 0.5)  # 80%
        validVideosCount = videoList.__len__() - trainVideosCount
        setValidVideos = validVideos(validVideosCount, videoList.__len__())

        # Раскадровка каждого видеофайла в каждой папке
        videoCount = 0
        for video in videoList:
            videoCount += 1

            # Создание пути сохренения раскадровки
            if (setValidVideos.__len__() == 0 or not setValidVideos.__contains__(videoCount)):
                save_path = save_root_directory_path + 'train/' + videoFolder + '/'
            else:
                save_path = save_root_directory_path + 'valid/' + videoFolder + '/'

            vp = video_path + video
            sp = save_path + 'video_' + str(videoCount)
            if not os.path.exists(sp):  # Если пути не существует создаем его
                os.makedirs(sp)

            # Раскадровка
            start = time.time()
            extractImages(vp, sp + '/')
            currentTime = time.time() - start

            # Обновление данных для статистики
            currentVideo += 1
            avgTime = (avgTime + currentTime) / 2

        currentFolder += 1
        print('*************************')
        print('Done:')
        print('\tFolders: ' + str(currentFolder) + '/' + str(totalFolders))
        print('\tVideos: ' + str(currentVideo) + '/' + str(totalVideos))
        print('\tTime left: ' + str(round(avgTime * (totalVideos - currentVideo) / 60, 1)) + ' minutes')

    print('*************************')



if __name__ == '__main__':
    video_root_directory_path = 'C:/neural-networks/datasets/UAVGesture/'
    save_root_directory_path = 'C:/neural-networks/datasets/TestUAVGesture/frames-short-70-cut-224-full/'

    totalStart = time.time()
    main(video_root_directory_path, save_root_directory_path)
    print('Total time: ' + str(round((time.time() - totalStart) / 60)) + ' minutes')
    print('Storyboard ended success!')
