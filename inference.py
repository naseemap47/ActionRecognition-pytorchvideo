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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Video
video = EncodedVideo.from_path('data/NonViolence/NV_23.mp4')
video_data = video.get_clip(0, 2)
video_data = video_transform(video_data)
print(video_data['video'].shape)

# model
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
