from typing import Optional
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPEncoderLayer
from torch import nn
import torch
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from huggingface_hub import PyTorchModelHubMixin


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


# model_to_hub = VitImageFeatureExtractor()
# model_to_hub.visual_projection =   other_model.feature_extractor.visual_projection
# model_to_hub.vision_model =        other_model.feature_extractor.vision_model
# model_to_hub.push_to_hub("ViT-bigG-14-laion2B-39B-b160k")
# model_from_hub = VitImageFeatureExtractor.from_pretrained("dbuos/ViT-bigG-14-laion2B-39B-b160k")


###########
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
import torch
from typing import Optional, Union, Tuple
from transformers.models.clip.modeling_clip import CLIPVisionConfig, CLIPEncoder
from torch import nn


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, embeddings, pre_layrnorm):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = embeddings
        self.pre_layrnorm = pre_layrnorm
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return pooled_output


def encoder_as_sequential(module_list):
    layer_list = [EncoderLayerSimple(layer) for layer in module_list]
    clip_encoder = nn.Sequential(*layer_list)


class EncoderLayerSimple(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.enc_layer = layer

    def forward(self, hidden_states: torch.Tensor) -> torch.FloatTensor:
        layer_outputs = self.enc_layer(hidden_states, None, None, output_attentions=False)
        hidden_states = layer_outputs[0]
        return hidden_states
