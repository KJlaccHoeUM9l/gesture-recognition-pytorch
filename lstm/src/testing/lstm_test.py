import argparse
import os
import cv2
import PIL.Image as Image
import numpy as np

from lstm.src.training.lstm_arch import *

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--frames_dir', default=None, help='video frames to classify (default: webcam)')
parser.add_argument('--freq', type=int, default=20, help='classify frequency (length of subsequence)')

num_classes = 9
classes = ['HoldHover', 'Land', 'LiftOff', 'MoveDownward', 
           'MoveForward', 'MoveLeft', 'MoveRight', 'MoveUpward', 'ReleaseSlingLoad']


def test(input, model):
    model.eval()
    input_var = Variable(input)
    input_var = input_var.cuda()
    outputs, _ = model(input_var)
    weight = Variable(torch.Tensor(range(outputs.shape[0])) / (outputs.shape[0] - 1) * 2).cuda()
    output = torch.mean(outputs * weight.unsqueeze(1), dim=0)
    output = nn.functional.softmax(output, dim=0)

    confidence, idx = torch.max(output.data.cpu(), 0)

    return confidence.numpy(), idx.numpy()


def main():
    model_info = torch.load(args.model)

    print('LSTM using pretrained model ' + model_info['arch'])
    print('Epochs ' + str(model_info['epoch']))

    # Load model
    original_model = models.__dict__[model_info['arch']](pretrained=False)
    model = FineTuneLstmModel(original_model, model_info['arch'],
            num_classes, model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
    model.cuda()
    model.load_state_dict(model_info['state_dict'])

    # data preprocessing...
    tran = transforms.Compose ([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.339, 0.224, 0.225])])

    if args.frames_dir is not None:
        # classify a video 
        frames_list = sorted(os.listdir(args.frames_dir))
        sublist = []
        for idx in range(len(frames_list)):
            sublist.append(frames_list[idx])
            
            if len(sublist) == args.freq and idx < len(frames_list)-1:
                frames = []
                for f in sublist:
                    frame = Image.open(args.frames_dir + '/' + f)
                    frame = tran(frame)
                    frames.append(frame)
                frames = torch.stack(frames)
                frames = frames[:-1] - frames[1:]

                print('classifying...')
                confidence, label = test(frames, model)
                print(classes[label], confidence)
                sublist = []
    else:
        # classify a web-camera
        # text settings
        text = None
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        # read frames from web-camera
        frames = []
        cap = cv2.VideoCapture(0)
        success, raw_frame = cap.read()
        while(success):
            # Our operations on the frame come here
            inputImage = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) # Потому что Image читает в BGR
            frame = tran(Image.fromarray(inputImage))
            frames.append(frame)

            if len(frames) == args.freq:
                frames = torch.stack(frames)
                frames = frames[:-1] - frames[1:]

                print('classifying...')
                confidence, label = test(frames, model)
                print(classes[label], confidence)

                if confidence > 0.9:
                    text = classes[label] + ": " + str(confidence)
                else:
                    text = None
                frames = []

            # Display the resulting frame
            cv2.putText(raw_frame, text, bottomLeftCornerOfText,
                        font, fontScale, fontColor, lineType)
            cv2.imshow('frame', raw_frame)
            cv2.imshow('input', np.array(frame[0]))

            success, raw_frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        print("stop")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()
    args.model = '../../weight/model_best_865.pth.tar'
    #args.frames_dir = 'C:/neural-networks/datasets/TestUAVGesture/frames-short-70-cut-224-part/train/Move To Right/video_2/'
    args.freq = 20

    print(args)
    main()
