from typing import Tuple

import torch
from torch import nn
from torch import optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchmetrics import Accuracy
from torchvision.models import ViT_L_16_Weights, vit_l_16
from torchvision.transforms import Normalize
from mlflow import log_metric
from statistics import mean



class VitFeatureExtractorCore(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.vit_model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.vit_model.heads = nn.Identity()
        self.vit_model.requires_grad_(False)

        self.transforms = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.denorm = Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1.0 / s for s in std],
        )

    def forward(self, x):
        img_a, img_b = x
        return self.vit_model(img_a), self.vit_model(img_b)

    def preprocess_input(self, x):
        return self.transforms(x)

    def reverse_preprocess(self, x):
        return (self.denorm(x) * 255).to(torch.uint8)


class VitFeatureExtractor(torch.nn.Module):
    def __init__(self, extractor=VitFeatureExtractorCore()):
        super().__init__()
        self.extractor = extractor
        self.transforms = extractor.transforms

    def forward(self, x):
        img_a, img_b = x
        return self.extractor((img_a, img_b))

    def reverse_preprocess(self, x):
        return self.extractor.reverse_preprocess(x)

# import pytorch_lightning as pl
# class LFullModel(pl.LightningModule):
#     def __init__(self, model, lr=1e-4, optimizer="Adam", save_checkpoint_fn=None):
#         super().__init__()
#         self.model = model
#         self.lr = lr
#         self.loss_fn = nn.BCEWithLogitsLoss()
#         self.train_acc = Accuracy(task="binary")
#         self.valid_acc = Accuracy(task="binary")
#         self.test_acc = Accuracy(task="binary")
#         self.to_inspect = None
#         self.on_sanity = False
#         self.optimizer = optimizer
#         self.best_valid_loss = 1e10
#         self.save_checkpoint_fn = save_checkpoint_fn
#
#     def forward(self, x):
#         return self.model(x)
#
#     def step(self, batch):
#         (img_a_batch, img_b_batch), label_batch = batch
#         y_true = label_batch.reshape(-1, 1).to(torch.float32)
#         y_pred = self.model((img_a_batch, img_b_batch))
#         loss = self.loss_fn(y_pred, y_true)
#         return loss, y_pred, y_true
#
#     def training_step(self, batch, batch_idx):
#         loss, y_pred, y_true = self.step(batch)
#         self.train_acc.update(y_pred, y_true.to(torch.uint8))
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss, y_pred, y_true = self.step(batch)
#         self.valid_acc.update(y_pred, y_true.to(torch.uint8))
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         loss, y_pred, y_true = self.step(batch)
#         self.test_acc.update(y_pred, y_true.to(torch.uint8))
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = getattr(optim, self.optimizer)(self.model.parameters(), lr=self.lr)
#         scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "train_loss",
#             }
#         }
#
#     def test_epoch_end(self, losses):
#         self.log_metrics(losses, self.test_acc, "test")
#
#     def training_epoch_end(self, losses):
#         self.log_metrics(losses, self.train_acc, "train")
#
#     def validation_epoch_end(self, losses):
#         avg_loss = self.log_metrics(losses, self.valid_acc, "valid")
#         if avg_loss is not None and avg_loss < self.best_valid_loss:
#             self.best_valid_loss = avg_loss
#             if self.save_checkpoint_fn is not None:
#                 self.save_checkpoint_fn(self.model)
#
#     def log_metrics(self, losses, accuracy: Accuracy, type):
#         if self.on_sanity:
#             print(f"Sanity check {type}")
#             return
#
#         if len(losses) == 0:
#             return
#
#         if len(losses) > 0 and isinstance(losses[0], torch.Tensor):
#             avg_loss = mean([x.item() for x in losses])
#         else:
#             avg_loss = mean([x["loss"].item() for x in losses])
#
#         avg_acc = accuracy.compute().item()
#         accuracy.reset()
#
#         log_metric(f"{type}_accuracy", avg_acc, step=self.current_epoch + 1)
#         log_metric(f"{type}_loss", avg_loss, step=self.current_epoch + 1)
#         self.log_dict(
#             {'loss': {f'{type}': avg_loss}, f"{type}_loss": avg_loss, 'accuracy': {f'{type}': avg_acc},
#              'step': self.current_epoch + 1.0})
#         return avg_loss


class ConcatFeatureDetector(torch.nn.Module):

    def __init__(self, features_dim=1024, hidden_dim=128):
        super().__init__()
        self.cls_layer = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(features_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features_a, features_b = x
        features = torch.cat((features_a, features_b), dim=1)
        return self.cls_layer(features)


class DiffFeatureDetector(torch.nn.Module):

    def __init__(self, features_dim=1024, hidden_dim=128):
        super().__init__()
        self.cls_layer = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features_a, features_b = x
        features = features_a - features_b
        return self.cls_layer(features)


class DiffFeatureDetectorParam(torch.nn.Module):

    def __init__(self, features_dim=1024, num_hidden=1, hidden_units=128, features_dropout=0.35, hidden_dropout=0.5,
                 hidden_act='relu'):
        super().__init__()
        self.cls_layer = nn.Sequential()
        self.cls_layer.add_module('features_dropout', nn.Dropout(features_dropout))
        for i in range(num_hidden):
            self.cls_layer.add_module(f'linear_{i}', nn.Linear(features_dim, hidden_units))
            if hidden_act == 'relu':
                self.cls_layer.add_module(f'relu_{i}', nn.ReLU())
            else:
                self.cls_layer.add_module(f'tanh_{i}', nn.Tanh())
            self.cls_layer.add_module(f'hidden_dropout_{i}', nn.Dropout(hidden_dropout))
            features_dim = hidden_units
        self.cls_layer.add_module('linear_out', nn.Linear(features_dim, 1))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        features_a, features_b = x
        features = features_a - features_b
        return self.cls_layer(features)


class DiffFeatureDetectorParamBiDirectional(DiffFeatureDetectorParam):

    def __init__(self, features_dim=1024, num_hidden=1, hidden_units=128, features_dropout=0.35, hidden_dropout=0.5,
                 hidden_act='relu'):
        super().__init__(features_dim, num_hidden, hidden_units, features_dropout, hidden_dropout, hidden_act)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        features_a, features_b = x
        features = features_a - features_b
        features2 = features_b - features_a
        return (self.cls_layer(features) + self.cls_layer(features2)) / 2.0


class SubModule(torch.nn.Module):
    def forward(self, a, b):
        return a - b


class CatModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=1)


class ProjectionReductionFeatureDetector(torch.nn.Module):

    def __init__(self, features_dim=1024, num_hidden=1, hidden_units=128, features_dropout=0.35, hidden_dropout=0.5,
                 projection_dim=512, merge_operation='subtract',
                 hidden_act='relu'):
        super().__init__()
        self.proj_layer = nn.Linear(features_dim, projection_dim)

        if merge_operation == 'subtract':
            self.merge_op = SubModule()
            features_dim = projection_dim
        else:
            self.merge_op = CatModule()
            features_dim = projection_dim * 2

        self.cls_layer = nn.Sequential()
        self.cls_layer.add_module('features_dropout', nn.Dropout(features_dropout))
        for i in range(num_hidden):
            self.cls_layer.add_module(f'linear_{i}', nn.Linear(features_dim, hidden_units))
            if hidden_act == 'relu':
                self.cls_layer.add_module(f'relu_{i}', nn.ReLU())
            else:
                self.cls_layer.add_module(f'tanh_{i}', nn.Tanh())
            self.cls_layer.add_module(f'hidden_dropout_{i}', nn.Dropout(hidden_dropout))
            features_dim = hidden_units
        self.cls_layer.add_module('linear_out', nn.Linear(hidden_units, 1))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        features_a, features_b = x
        a = self.proj_layer(features_a)
        b = self.proj_layer(features_b)
        features_0 = self.merge_op(a, b)
        features_1 = self.merge_op(b, a)
        logits_0 = self.cls_layer(features_0)
        logits_1 = self.cls_layer(features_1)
        return (logits_0 + logits_1) / 2.0


# def create_lit_fmodel():
#     return LFullModel(ConcatFeatureDetector())
#
#
# def create_lit_fmodel_diff():
#     return LFullModel(DiffFeatureDetector())
