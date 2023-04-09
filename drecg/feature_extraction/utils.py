from pathlib import Path

from tqdm.auto import tqdm
import torch
from transformers import AutoProcessor, AutoModel
from drecg.data.utils import create_dataloader_train, create_dataloader_test, create_dataloader_validation
from drecg.models.feat_extraction import VitFeatureExtractor
from torchvision.transforms import ToPILImage, Compose, RandomHorizontalFlip, RandomRotation, RandomAdjustSharpness, \
    RandomAutocontrast, ColorJitter


def extract_features(dataloader, feat_extractor, device, data=None):
    if data is None:
        data = []
    bar = tqdm(range(len(dataloader)))
    feat_extractor.to(device)
    feat_extractor.eval()
    with torch.no_grad():
        for x_batch, y_batch, paths in dataloader:
            x_batch = x_batch[0].to(device), x_batch[1].to(device)
            features_a, features_b = feat_extractor(x_batch)

            features_a = features_a.detach().cpu()
            features_b = features_b.detach().cpu()
            features = (features_a, features_b)

            torch.cuda.empty_cache()
            data.append((features, y_batch, paths))
            bar.update(1)
    return data


class VitLaionPreProcess(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    def forward(self, img):
        out = self.processor(images=img, return_tensors="pt")
        return out.data['pixel_values'].squeeze()


class VitLaionFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_model = AutoModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        self.transforms = VitLaionPreProcess()

    def forward(self, x):
        img_a, img_b = x
        return self.vit_model.get_image_features(pixel_values=img_a), self.vit_model.get_image_features(
            pixel_values=img_b)


class VitFeatureExtractorComplete(VitLaionFeatureExtractor):
    def __init__(self, output_attentions=None, output_hidden_states=None, use_return_dict=None):
        super().__init__()
        self.output_attentions = output_attentions if output_attentions is not None else self.vit_model.config.output_attentions
        self.output_hidden_states = output_hidden_states if output_hidden_states is not None else self.vit_model.config.output_hidden_states
        self.use_return_dict = use_return_dict if use_return_dict is not None else self.vit_model.config.use_return_dict

    def forward(self, x):
        img_a, img_b = x
        return self.forward_single(img_a), self.forward_single(img_b)

    def forward_single(self, pixel_values, output_attentions=None, output_hidden_states=None, use_return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        use_return_dict = use_return_dict if use_return_dict is not None else self.use_return_dict
        vision_outputs = self.vit_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=use_return_dict,
        )
        return vision_outputs.last_hidden_state


class UFormV1FeatureExtractor(torch.nn.Module):
    def __init__(self, model=None, model_name='unum-cloud/uform-vl-english'):
        super().__init__()
        if model is not None:
            self.vit_model = model
        else:
            from drecg.models.uform import get_model
            self.vit_model = get_model(model_name)
        self.transforms = self.vit_model.preprocess_image

    def forward(self, x):
        img_a, img_b = x
        return self.vit_model.encode_image(img_a), self.vit_model.encode_image(img_b)


def create_augmentation_transforms(final_transforms):
    return Compose([
        ColorJitter(brightness=0.3, saturation=0.1, hue=0.05),
        RandomAutocontrast(p=0.9),
        RandomHorizontalFlip(p=0.9),
        RandomRotation(degrees=10),
        RandomAdjustSharpness(sharpness_factor=0.5, p=0.9),
        final_transforms
    ])


def extract_features_with_model(model='ViT_L_16', root_dir='features_ext_vit'):
    import gc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model == 'ViT_L_16':
        feat_extractor = VitFeatureExtractor()
    elif model == 'ViT_LAION':
        feat_extractor = VitLaionFeatureExtractor()
    elif model == 'ViT_LAION_COMPLETE':
        feat_extractor = VitFeatureExtractorComplete()
    elif model == 'UForm_V1':
        feat_extractor = UFormV1FeatureExtractor()
    else:
        raise ValueError(f'Model {model} not found')

    train_transform = create_augmentation_transforms(feat_extractor.transforms)

    train_dataloader = create_dataloader_train(transforms=feat_extractor.transforms)
    train_augmented_dataloader = create_dataloader_train(transforms=train_transform)
    test_dataloader = create_dataloader_test(transforms=feat_extractor.transforms)
    validation_dataloader = create_dataloader_validation(transforms=feat_extractor.transforms)

    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    features = extract_features(test_dataloader, feat_extractor, device=device)
    torch.save(features, f'{root_dir}/test_features.pt')

    features = extract_features(validation_dataloader, feat_extractor, device=device)
    torch.save(features, f'{root_dir}/validation_features.pt')

    del features
    gc.collect()

    features = extract_features(train_dataloader, feat_extractor, device=device)
    torch.save(features, f'{root_dir}/train_features.pt')

    del features
    gc.collect()

    features = extract_features(train_augmented_dataloader, feat_extractor, device=device)
    torch.save(features, f'{root_dir}/train_features_augmented_p0.pt')

    del features
    gc.collect()

    features = extract_features(train_augmented_dataloader, feat_extractor, device=device)
    torch.save(features, f'{root_dir}/train_features_augmented_p1.pt')







