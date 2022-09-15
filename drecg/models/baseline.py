import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import Accuracy
from torch.optim import Adam


class LFullModel(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        (img_a_batch, img_b_batch), label_batch = batch
        y_true = label_batch.reshape(-1, 1).to(torch.float32)
        y_pred = self.model((img_a_batch, img_b_batch))
        loss = self.loss_fn(y_pred, y_true)
        return loss, y_pred, y_true    

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self.step(batch)
        self.train_acc(y_pred, y_true.to(torch.uint8))
        self.log_dict({'train_loss': loss, 'train_acc': self.train_acc}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self.step(batch)
        self.valid_acc(y_pred, y_true.to(torch.uint8))
        self.log_dict({'valid_loss': loss, 'valid_acc': self.valid_acc}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)


class FeatureExtractorConv2d(torch.nn.Module):

    def __init__(self, out_base = 16):
        super().__init__()
        self.sequential = nn.Sequential(
            self.create_conv_module(in_channels=3, out_channels=out_base, kernel_size=3),
            self.create_conv_module(in_channels=out_base, out_channels=out_base*2, kernel_size=3), 
            self.create_conv_module(in_channels=out_base*2, out_channels=out_base*4, kernel_size=3),
            self.create_conv_module(in_channels=out_base*4, out_channels=out_base*8, kernel_size=3),
            self.create_conv_module(in_channels=out_base*8, out_channels=out_base*8, kernel_size=3),
            self.create_conv_module(in_channels=out_base*8, out_channels=out_base*16, kernel_size=3),
            nn.MaxPool2d(kernel_size=8),
            nn.Flatten()
        )

    @staticmethod
    def create_conv_module(in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
        module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        )
        if batch_norm:
            module.add_module('batch_norm', nn.BatchNorm2d(num_features=out_channels))
        module.add_module('relu', nn.ReLU())
        module.add_module('max_pool', nn.MaxPool2d(kernel_size=2))

        return module 

    def forward(self, x):
        return self.sequential(x)



class Detector(torch.nn.Module):

    def __init__(self, feature_extractor=FeatureExtractorConv2d, features_dim=256, hidden_dim=128, train_extractor=True):
        super().__init__()
        self.extractor_a = feature_extractor()
        self.extractor_b = feature_extractor()
        self.train_extractor = train_extractor
        self.cls_layer = nn.Sequential(
            nn.Linear(features_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse) if self.train_extractor else self.cls_layer.parameters(recurse)    

    def forward(self, x):
        img_a, img_b = x
        if self.train_extractor:
            features_a = self.extractor_a(img_a)
            features_b = self.extractor_b(img_b)
        else:
            with torch.no_grad():
                features_a = self.extractor_a(img_a)
                features_b = self.extractor_b(img_b)        
        feature = torch.cat((features_a, features_b), dim=1)
        return self.cls_layer(feature)
