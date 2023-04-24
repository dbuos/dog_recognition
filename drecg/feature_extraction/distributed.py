from typing import Optional

from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from torch import nn
import torch
from huggingface_hub import PyTorchModelHubMixin


class EncoderLayerSimple(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.enc_layer = layer

    def forward(self, hidden_states: torch.Tensor) -> torch.FloatTensor:
        layer_outputs = self.enc_layer(hidden_states, None, None, output_attentions=False)
        hidden_states = layer_outputs[0]
        return hidden_states


def encoder_as_list(module_list):
    layer_list = [EncoderLayerSimple(layer) for layer in module_list]
    return layer_list

class EncoderPostLayer(nn.Module):
    def __init__(self, post_layernorm):
        super().__init__()
        self.post_layernorm = post_layernorm

    def forward(self, last_hidden_state):
        pooled_output = last_hidden_state[:, 0, :]
        return self.post_layernorm(pooled_output)


class VitImageFeatureExtractor(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()

        config = {
            "attention_dropout": 0.0,
            "dropout": 0.0,
            "hidden_act": "gelu",
            "hidden_size": 1664,
            "image_size": 224,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "layer_norm_eps": 1e-05,
            "model_type": "clip_vision_model",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 48,
            "patch_size": 14,
            "projection_dim": 512,
            "transformers_version": "4.27.2"
        }
        vision_config = CLIPVisionConfig(**config)

        self.projection_dim = 1280
        self.vision_embed_dim = 1664

        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)

    def forward(self, x):
        return self.get_image_features(pixel_values=x)

    def get_image_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> torch.FloatTensor:
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    @classmethod
    def load_pretrained(cls, name="dbuos/ViT-bigG-14-laion2B-39B-b160k"):
        return cls.from_pretrained(name)

    def to_sequential(self):
        """Converts the model to a sequential model in order to use in Pipeline parallelism"""
        modules_list = []

        emb_and_pre_layrnorm = nn.Sequential(self.vision_model.embeddings, self.vision_model.pre_layrnorm)
        modules_list.append(emb_and_pre_layrnorm)

        encoder_modules = encoder_as_list(self.vision_model.encoder.layers)
        modules_list.extend(encoder_modules)

        encoder_post_layer = EncoderPostLayer(self.vision_model.post_layernorm)
        modules_list.append(encoder_post_layer)

        modules_list.append(self.visual_projection)

        return nn.Sequential(*modules_list)


def sequential_model_to_devices(model, device0, device1, device2, device3):
    device0_layers = len(model) // 4
    for i, module in enumerate(model):
        if i < device0_layers:
            module.to(device0)
        elif i < device0_layers*2:
            module.to(device1)
        elif i < device0_layers*3:
            module.to(device2)
        else:
            module.to(device3)


# model_to_hub = VitImageFeatureExtractor()
# model_to_hub.visual_projection =   other_model.feature_extractor.visual_projection
# model_to_hub.vision_model =        other_model.feature_extractor.vision_model
# model_to_hub.push_to_hub("ViT-bigG-14-laion2B-39B-b160k")
# model_from_hub = VitImageFeatureExtractor.from_pretrained("dbuos/ViT-bigG-14-laion2B-39B-b160k")

