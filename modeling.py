
#### modeling.py
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import torch
# import dinowrapper
from lrm.models.encoders.dino_wrapper2 import DinoWrapper
from lrm.models.transformer import TriplaneTransformer
from lrm.models.rendering.synthesizer_part import TriplaneSynthesizer

class CameraEmbedder(nn.Module):
    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)

class LRMGeneratorConfig(PretrainedConfig):
    model_type = "lrm_generator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera_embed_dim = kwargs.get("camera_embed_dim", 1024)
        self.rendering_samples_per_ray = kwargs.get("rendering_samples_per_ray", 128)
        self.transformer_dim = kwargs.get("transformer_dim", 1024)
        self.transformer_layers = kwargs.get("transformer_layers", 16)
        self.transformer_heads = kwargs.get("transformer_heads", 16)
        self.triplane_low_res = kwargs.get("triplane_low_res", 32)
        self.triplane_high_res = kwargs.get("triplane_high_res", 64)
        self.triplane_dim = kwargs.get("triplane_dim", 80)
        self.encoder_freeze = kwargs.get("encoder_freeze", False)
        self.encoder_model_name = kwargs.get("encoder_model_name", 'facebook/dinov2-base')
        self.encoder_feat_dim = kwargs.get("encoder_feat_dim", 768)

class LRMGenerator(PreTrainedModel):
    config_class = LRMGeneratorConfig

    def __init__(self, config: LRMGeneratorConfig):
        super().__init__(config)

        self.encoder_feat_dim = config.encoder_feat_dim
        self.camera_embed_dim = config.camera_embed_dim

        self.encoder = DinoWrapper(
            model_name=config.encoder_model_name,
            freeze=config.encoder_freeze,
        )
        self.camera_embedder = CameraEmbedder(
            raw_dim=12 + 4, embed_dim=config.camera_embed_dim,
        )
        self.transformer = TriplaneTransformer(
            inner_dim=config.transformer_dim, num_layers=config.transformer_layers, num_heads=config.transformer_heads,
            image_feat_dim=config.encoder_feat_dim,
            camera_embed_dim=config.camera_embed_dim,
            triplane_low_res=config.triplane_low_res, triplane_high_res=config.triplane_high_res, triplane_dim=config.triplane_dim,
        )
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=config.triplane_dim, samples_per_ray=config.rendering_samples_per_ray,
        )

    def forward(self, image, camera):
        assert image.shape[0] == camera.shape[0], "Batch size mismatch"
        N = image.shape[0]

        # encode image
        image_feats = self.encoder(image)
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        # embed camera
        camera_embeddings = self.camera_embedder(camera)
        assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
            f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        # transformer generating planes
        planes = self.transformer(image_feats, camera_embeddings)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"
        return planes
