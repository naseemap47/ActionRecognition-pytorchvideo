import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# pytorchvideo
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    Permute
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
import torchmetrics


# Load Dataset
non_voi = glob.glob('data/NonViolence/*.mp4')
voi = glob.glob('data/Violence/*.mp4')
labels = [0] * len(non_voi) + [1] * len(voi)

df = pd.DataFrame(zip(non_voi+voi, labels), columns=['file', 'label'])
# print(df.head())

# Train Val split
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)

# Train Test split
# train_df, test_df = train_test_split(train_df, test_size=0.1, shuffle=True)

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

# Train Dataset
# train_dataset = labeled_video_dataset(
#     train_df, clip_sampler=make_clip_sampler('random', 2),
#     transform=video_transform, decode_audio=False
# )

# loader = DataLoader(train_dataset, batch_size=5, num_workers=0, pin_memory=True)

# batch = next(iter(loader))
# print(batch.keys())
# print(batch['video'].shape, batch['label'].shape)




class OurModel(LightningModule):
    def __init__(self):
        super(OurModel, self).__init__()

        # Model architecute
        self.video_model = torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(400, 1)

        self.lr = 1e-3
        self.batch_size = 4
        self.numworker = 4

        # Evaluation Metric
        self.metric = torchmetrics.Accuracy(task='binary')

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.video_model(x)
        x = self.relu(x)
        x = self.linear(x)
        return x
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {'optimizer':opt, 'lr_scheduler': scheduler}

    def train_dataloader(self):
        dataset = labeled_video_dataset(
            train_df, clip_sampler=make_clip_sampler('random', 2),
            transform=video_transform, decode_audio=False
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.numworker, pin_memory=True)
        return loader

    def training_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        out =  self.forward(video)
        loss = self.criterion(out, label)
        metric = self.metric(out, label.to(torch.int64))
        return {'loss': loss, 'metric': metric.detach()}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        metric = torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        self.log('training_loss', loss)
        self.log('training_metric', metric)

    # Validation
    def val_dataloader(self):
        dataset = labeled_video_dataset(
            val_df, clip_sampler=make_clip_sampler('random', 2),
            transform=video_transform, decode_audio=False
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.numworker, pin_memory=True)
        return loader

    def validation_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        out =  self.forward(video)
        loss = self.criterion(out, label)
        metric = self.metric(out, label.to(torch.int64))
        return {'loss': loss, 'metric': metric.detach()}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        metric = torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        self.log('val_loss', loss)
        self.log('val_metric', metric)

    # Test data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    def test_dataloader(self):
        dataset = labeled_video_dataset(
            val_df, clip_sampler=make_clip_sampler('random', 2),
            transform=video_transform, decode_audio=False
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.numworker, pin_memory=True)
        return loader

    def test_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        out =  self.forward(video)
        return {'label': label.detach(), 'pred': out.detach()}

    def test_epoch_end(self, outputs):
        label = torch.cat([x['label'] for x in outputs]).cpu().numpy()
        pred = torch.cat([x['pred'] for x in outputs]).cpu().numpy()
        pred = np.where(pred>0.5, 1, 0)
        print(classification_report(label, pred))
        

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath='checkpoints',
    filename='best', save_last=True
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

model = OurModel()
seed_everything(0)

trainer = Trainer(
    max_epochs=15, accelerator='gpu', devices=-1,
    precision=16, accumulate_grad_batches=2,
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    callbacks=[lr_monitor, checkpoint_callback]
)

# train custom model
trainer.fit(model)

# Save Custom Model
torch.save(model.state_dict(), 'model.pt')

# Validation
print(trainer.validate(model))

# Testing
print(trainer.test(model))
