"""ViT encoder + autoregressive Transformer IDS decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VisionIDSModel(nn.Module):
    """Decode prefix IDS tokens from one cropped glyph image.

    The default timm model is a DINOv2-initialized ViT-S/14. Set
    ``pretrained=False`` for offline tests that must not download weights.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        encoder_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        encoder_checkpoint: str | None = None,
        decoder_dim: int = 384,
        decoder_layers: int = 6,
        decoder_heads: int = 6,
        max_length: int = 128,
        image_size: int = 224,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("训练模型需要安装 `pip install -e '.[train]'`") from exc

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained and not encoder_checkpoint,
            num_classes=0,
            global_pool="",
            img_size=image_size,
            checkpoint_path=encoder_checkpoint or "",
        )
        encoder_dim = int(self.encoder.num_features)
        self.memory_projection = nn.Linear(encoder_dim, decoder_dim)
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim)
        self.position_embedding = nn.Embedding(max_length, decoder_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.output = nn.Linear(decoder_dim, vocab_size)
        self.max_length = max_length

    def encode_image(self, images: Tensor) -> Tensor:
        features = self.encoder.forward_features(images)
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "x_prenorm", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
            else:
                raise RuntimeError(f"无法识别编码器输出字段：{tuple(features)}")
        if features.ndim == 2:
            features = features.unsqueeze(1)
        return self.memory_projection(features)

    def forward(self, images: Tensor, input_ids: Tensor) -> Tensor:
        if input_ids.shape[1] > self.max_length:
            raise ValueError(f"序列长度超过 max_length={self.max_length}")
        memory = self.encode_image(images)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        target = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                input_ids.shape[1],
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        decoded = self.decoder(target, memory, tgt_mask=causal_mask)
        return self.output(decoded)
