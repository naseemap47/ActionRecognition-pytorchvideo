import torch
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    Permute
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
from model import OurModel

# Augumentation
video_transform = Compose([
    ApplyTransformToKey(key='video',
    transform=Compose([
        UniformTemporalSubsample(20),
        Lambda(lambda x:x/255),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        RandomShortSideScale(min_size=248, max_size=256),
        CenterCropVideo(224),
        RandomHorizontalFlip(p=0.5),
    ])
    )
])

print('#################################################################')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Video
video = EncodedVideo.from_path('data/NonViolence/NV_10.mp4')
video_data = video.get_clip(0, 2)
print(np.shape(video_data['video']))
print(type(video_data['video']))
print(np.shape(video_data['audio']))
print(video_data['audio'])

print(video_data)

video_data = video_transform(video_data)
# print(video_data['video'].shape)
print(video_data.keys)
print(type(video_data))

# # # model
model = OurModel()
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint)
model.eval()
# print(model.eval())

model = model.cuda()
inputs = video_data['video'].cuda()
inputs = torch.unsqueeze(inputs, 0)
print(inputs.shape)

preds = model(inputs)
preds = preds.detach().cpu().numpy()
print(preds)

preds = np.where(preds>0.5, 1, 0)
print(preds)
print(preds[0][0])

print('#################################################################')

from collections import deque
import cv2
SEQUENCE_LENGTH = 60

frames_queue = deque(maxlen=SEQUENCE_LENGTH)

# empty_array = np.arange()
cap = cv2.VideoCapture('data/NonViolence/NV_10.mp4')

original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Write Video
out_vid = cv2.VideoWriter('output.mp4', 
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (original_video_width, original_video_height))
    
while True:
    success, img = cap.read()
    if not success:
        break
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames_queue.append(img_rgb)
    # torch.Size([3, 60, 720, 1280])
    if len(frames_queue) == SEQUENCE_LENGTH:
        input_img = np.reshape(frames_queue, (3, SEQUENCE_LENGTH, h, w))
        print(np.shape(input_img))
        input_tensor = torch.from_numpy(input_img).to(dtype=torch.float32)
        print(np.shape(input_tensor))
        input_dic = {
            'video': input_tensor,
            'audio': None
        }
        
        print(input_dic)
        print(type(input_dic))

        print(np.shape(input_dic['video']))
        video_data = video_transform(input_dic)
        print(np.shape(video_data['video']))

        inputs = video_data['video'].cuda()
        inputs = torch.unsqueeze(inputs, 0)
        print(inputs.shape)

        preds = model(inputs)
        preds = preds.detach().cpu().numpy()
        print(preds)

        preds = np.where(preds>0.5, 1, 0)
        print(preds)
        print(preds[0][0])

    #     cv2.putText(img, f'{preds[0][0]}', (50, 60),
    #                 cv2.FONT_HERSHEY_PLAIN, 2,
    #                 (0, 255, 0), 2

    #     )
    # cv2.imshow('img', img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break

print('#################################################################')
