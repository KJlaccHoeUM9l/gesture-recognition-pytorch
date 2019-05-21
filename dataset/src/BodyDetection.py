import cv2
import time


def getRectangle(videoPath, maxFrames, flagResize=False, width=480, height=270):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    avg_x = 0
    avg_y = 0
    avg_w = 0
    avg_h = 0
    total = 0
    frameCount = 0

    cap = cv2.VideoCapture(videoPath)
    success, frame = cap.read()
    while success and frameCount < maxFrames:
        if flagResize:
            frame = cv2.resize(frame, (width, height))

        rectangle, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
        if len(rectangle) != 0:
            avg_x += rectangle[0][0]
            avg_y += rectangle[0][1]
            avg_w += rectangle[0][2]
            avg_h += rectangle[0][3]
            total += 1

        success, frame = cap.read()
        frameCount += 1

    avg_x = int(avg_x / total)
    avg_y = int(avg_y / total)
    avg_w = int(avg_w / total)
    avg_h = int(avg_h / total)

    return avg_x, avg_y, avg_w, avg_h

def getSquare(x, y, w, h):
    largestEdge = max(w, h)
    shift = int(abs(w - h) / 2.0)
    if largestEdge == w:
        y -= shift
        h = largestEdge
    else:
        x -= shift
        w = largestEdge
    return x, y, w, h

def cut_image(img, x, y, w, h, newSize):
    return cv2.resize(img[y:y+h, x:x+w], (newSize, newSize))


if __name__ == '__main__':
    maxFrames = 50
    flagResize = True
    videoPath = 'C:/neural-networks/datasets/UAVGesture/Slow Down/S6_slowDown_HD.mp4'
    #videoPath = 'C:/Users/пк/Desktop/S12_allClear_HD.mp4'

    totalStart = time.time()

    x, y, w, h = getRectangle(videoPath, maxFrames, flagResize)
    x, y, w, h = getSquare(x, y, w, h)
    frameCount = 0

    cap = cv2.VideoCapture(videoPath)
    success, frame = cap.read()
    while success and frameCount < maxFrames:
        frame = cv2.resize(frame, (480, 270))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #frame = cut_image(frame, x, y, w, h, 224)
        cv2.imshow('feed', frame)

        success, frame = cap.read()
        frameCount += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Total time: ' + str(round((time.time() - totalStart))) + ' sec')

    cap.release()
    cv2.destroyAllWindows()
