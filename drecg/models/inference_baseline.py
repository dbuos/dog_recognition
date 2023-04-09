
import torch
from drecg.models.feat_extraction import VitFeatureExtractorCore
from drecg.data.utils import create_dataloader

from torchmetrics import Accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VitInferenceBaseline(torch.nn.Module):
    def __init__(self, pretrain_cls_path):
        super(VitInferenceBaseline, self).__init__()
        self.feat_extractor = VitFeatureExtractorCore()
        self.classifier = torch.load(pretrain_cls_path)
        
        
    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.classifier(x)
        return x

    def evaluate(self, data_dir, batch_size=32, num_workers=4):
        self.eval()
        test_loader = create_dataloader(data_dir, batch_size=batch_size, num_workers=num_workers, transforms=self.feat_extractor.transforms, shuffle=False)
        acc = Accuracy()
        with torch.no_grad():
            for (x1, x2), y in test_loader:
                x = (x1.to(device), x2.to(device))
                y_pred = self(x).detach().cpu()
                y_pred = torch.round(torch.sigmoid(y_pred)).view_as(y).to(y.dtype)
                acc.update(y_pred, y)
        return acc.compute().item()
