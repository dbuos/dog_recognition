import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import Accuracy
from torch.optim import Adam
from torchvision.models import ViT_L_16_Weights, vit_l_16
from torchvision.transforms import Normalize
from mlflow import log_metric, log_param, log_artifacts
from statistics import mean


class VitFeatureExtractor(torch.nn.Module):
    def __init__(self, device):
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

        self.device = device
        self.vit_model.to(device)

    def forward(self, x):
        img_a, img_b = x
        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        return self.vit_model(img_a), self.vit_model(img_b)

    def preprocess_input(self, x):
        return self.transforms(x)    

    def reverse_preprocess(self, x):
        return (self.denorm(x)*255).to(torch.uint8)


class LFullModel(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()
        self.to_inspect = None
        self.on_sanity = False


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
        self.train_acc.update(y_pred, y_true.to(torch.uint8))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self.step(batch)
        self.valid_acc.update(y_pred, y_true.to(torch.uint8))
        return loss

    def test_step(self, batch, batch_idx):
         loss, y_pred, y_true = self.step(batch)
         self.test_acc.update(y_pred, y_true.to(torch.uint8))
         return loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
    
    def test_epoch_end(self, losses):
        self.log_metrics(losses, self.test_acc, "test")

    def training_epoch_end(self, losses):
        self.log_metrics(losses, self.train_acc, "train")
   
    def validation_epoch_end(self, losses):
        self.log_metrics(losses, self.valid_acc, "valid")

    def log_metrics(self, losses, accuracy: Accuracy, type):
        if self.on_sanity:
            print(f"Sanity check {type}")
            return

        if len(losses) == 0:
            return    

        if len(losses) > 0 and isinstance(losses[0], torch.Tensor):
            avg_loss = mean([x.item() for x in losses])
        else:
            avg_loss = mean([x["loss"].item() for x in losses])

        avg_acc = accuracy.compute().item()
        accuracy.reset()

        log_metric(f"{type}_accuracy", avg_acc, step=self.current_epoch+1)
        log_metric(f"{type}_loss", avg_loss, step=self.current_epoch+1)
        self.log_dict({'loss': {f'{type}': avg_loss}, 'accuracy': {f'{type}': avg_acc}, 'step': self.current_epoch+1.0})


class ConcatFeatureDetector(torch.nn.Module):

    def __init__(self, features_dim=1024, hidden_dim=128):
        super().__init__()
        self.cls_layer = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(features_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features_a, features_b = x
        features= torch.cat((features_a, features_b), dim=1)
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
        features= features_a - features_b
        return self.cls_layer(features)        

def create_lit_fmodel():
    return LFullModel(ConcatFeatureDetector())

def create_lit_fmodel_diff():
    return LFullModel(DiffFeatureDetector())    