import cv2


def get_rectangle(video_path, max_frames, width=480, height=270):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    avg_x = 0
    avg_y = 0
    avg_w = 0
    avg_h = 0
    total = 0
    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    while success and frame_count < max_frames:
        frame = cv2.resize(frame, (width, height))

        rectangle, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
        if len(rectangle) != 0:
            avg_x += rectangle[0][0]
            avg_y += rectangle[0][1]
            avg_w += rectangle[0][2]
            avg_h += rectangle[0][3]
            total += 1

        success, frame = cap.read()
        frame_count += 1

    avg_x = int(avg_x / total)
    avg_y = int(avg_y / total)
    avg_w = int(avg_w / total)
    avg_h = int(avg_h / total)

    return avg_x, avg_y, avg_w, avg_h


def get_square(x, y, w, h):
    largest_edge = max(w, h)
    shift = int(abs(w - h) / 2.0)
    if largest_edge == w:
        y -= shift
        h = largest_edge
    else:
        x -= shift
        w = largest_edge
    return x, y, w, h


def cut_image(img, x, y, w, h, new_size):
    return cv2.resize(img[y:y+h, x:x+w], (new_size, new_size))


def demonstration():
    video_path = 'C:/neural-networks/datasets/UAVGesture/Slow Down/S6_slowDown_HD.mp4'
    max_frames = 50

    x, y, w, h = get_rectangle(video_path, max_frames)
    x, y, w, h = get_square(x, y, w, h)
    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    while success and frame_count < max_frames:
        frame = cv2.resize(frame, (480, 270))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow('feed', frame)

        success, frame = cap.read()
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demonstration()
