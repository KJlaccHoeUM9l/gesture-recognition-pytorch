import cv2
import numpy as np
import torch
import PIL.Image as Image

from torchvision import transforms
from src.opts import parse_opts
from src.model import generate_model
from src.mean import get_mean, get_std
from src.spatial_transforms import Scale, CenterCrop


# UAV Gesture
num_classes = 13
classes = ['All clear', 'Have Command', 'Hover', 'Land', 'Landing Direction', 'Move Ahead', 'Move Downward',
           'Move To Left', 'Move To Right', 'Move Upward', 'Not Clear', 'Slow Down', 'Wave Off']


def test(input_frames, model):
    model.eval()

    input_frames = input_frames.unsqueeze(0)
    output = model(input_frames)
    confidence, idx = torch.max(output[0], 0)

    return confidence.item(), idx.item()


def main():
    opt = parse_opts()

    print('Using model: ' + opt.model)
    model, _ = generate_model(opt)
    model_info = torch.load(opt.trained_model_path)
    model.load_state_dict(model_info['state_dict'])

    # Data preprocessing
    if not opt.no_mean_norm:
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value, dataset=opt.std_dataset)
        norm_method = transforms.Normalize(mean=opt.mean, std=opt.std)
    else:
        norm_method = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

    transform = transforms.Compose([Scale(opt.sample_size),
                                    CenterCrop(opt.sample_size),
                                    transforms.ToTensor(),
                                    norm_method])

    # Text settings
    text = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    # Read frames from web-camera
    frames = []
    cap = cv2.VideoCapture(0)
    success, raw_frame = cap.read()
    while success:
        # Our operations on the frame come here
        frame = transform(Image.fromarray(raw_frame))
        frames.append(frame)

        if len(frames) == opt.sample_duration:
            frames = torch.stack(frames, 0).permute(1, 0, 2, 3)

            print('\tClassifying:')
            confidence, label = test(frames, model)
            print('\t\t', classes[label], confidence)

            if confidence > 0.8:
                text = classes[label] + ": " + str(confidence)
            else:
                text = None
            frames = []

        # Display the resulting frame
        cv2.putText(raw_frame, text, bottom_left_corner_of_text,
                    font, font_scale, font_color, line_type)
        cv2.imshow('frame', raw_frame)
        frame = frame.transpose(1, 0)
        frame = frame.transpose(2, 1)
        cv2.imshow('input', np.array(frame))

        success, raw_frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    print("stop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
