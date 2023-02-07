import torch.nn as nn
import torch
from pytorch_lightning import LightningModule


class OurModel(LightningModule):
    def __init__(self):
        super(OurModel, self).__init__()

        # Model architecute
        self.video_model = torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(400, 1)

    def forward(self, x):
        x = self.video_model(x)
        x = self.relu(x)
        x = self.linear(x)
        return x
