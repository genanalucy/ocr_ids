"""Checkpoint loading and greedy decoding for image-to-IDS inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from .ids import parse_ids, validate_ids
from .models import VisionIDSModel
from .preprocess import PreprocessResult, normalize_single_glyph
from .structure_scope import FLAT_STRUCTURE_OPERATORS
from .vocab import Vocabulary


@dataclass(frozen=True, slots=True)
class Prediction:
    ids: str
    confidence: float
    syntax_valid: bool
    tree: str | dict[str, object] | None
    tokens: tuple[str, ...]


class IDSInferencer:
    """Load a training checkpoint and greedily decode an IDS from a glyph image."""

    def __init__(self, checkpoint_path: str | Path, *, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"未找到 checkpoint：{self.checkpoint_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint:
            raise ValueError("checkpoint 缺少 model 或 config 字段")
        config = checkpoint["config"]
        self.flat_structure_only = config.get("task", {}).get("scope") == "flat-structure-v1"
        vocab_path = self.checkpoint_path.parent / "vocab.json"
        self.vocabulary = Vocabulary.load(vocab_path)
        model_config = config["model"]
        self.image_size = int(config["data"]["image_size"])
        self.model = VisionIDSModel(
            len(self.vocabulary.tokens),
            encoder_name=model_config["encoder_name"],
            pretrained=False,
            encoder_checkpoint=None,
            decoder_dim=int(model_config["decoder_dim"]),
            decoder_layers=int(model_config["decoder_layers"]),
            decoder_heads=int(model_config["decoder_heads"]),
            max_length=int(model_config["max_length"]),
            image_size=self.image_size,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self._transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare(self, image: Image.Image) -> PreprocessResult:
        return normalize_single_glyph(image, size=self.image_size)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        return self.predict_prepared(self.prepare(image).image)

    @torch.inference_mode()
    def predict_prepared(self, image: Image.Image) -> Prediction:
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        token_to_id = self.vocabulary.token_to_id
        ids_to_token = self.vocabulary.tokens
        generated = [token_to_id["<bos>"]]
        log_probabilities: list[float] = []
        blocked = (token_to_id["<pad>"], token_to_id["<bos>"])

        for _ in range(self.model.max_length - 1):
            inputs = torch.tensor([generated], dtype=torch.long, device=self.device)
            logits = self.model(tensor, inputs)[:, -1, :]
            logits[:, list(blocked)] = -torch.inf
            if self.flat_structure_only:
                emitted = len(generated) - 1
                if emitted == 0:
                    allowed = [token_to_id[token] for token in FLAT_STRUCTURE_OPERATORS if token in token_to_id]
                else:
                    root = ids_to_token[generated[1]]
                    arity = 3 if root in {"⿲", "⿳"} else 2
                    if emitted <= arity:
                        allowed = [
                            index
                            for index, token in enumerate(ids_to_token)
                            if token not in {"<pad>", "<bos>", "<eos>"}
                            and token not in FLAT_STRUCTURE_OPERATORS
                        ]
                    else:
                        allowed = [token_to_id["<eos>"]]
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[:, allowed] = False
                logits.masked_fill_(mask, -torch.inf)
            probabilities = torch.softmax(logits, dim=-1)
            next_id = int(torch.argmax(probabilities, dim=-1).item())
            log_probabilities.append(float(torch.log(probabilities[0, next_id]).item()))
            if next_id == token_to_id["<eos>"]:
                break
            generated.append(next_id)

        tokens = tuple(ids_to_token[item] for item in generated[1:])
        ids = "".join(tokens)
        problems = validate_ids(ids, strict_terminals=True)
        tree: str | dict[str, object] | None = None
        if not problems:
            tree = parse_ids(ids).to_dict()
        confidence = math.exp(sum(log_probabilities) / len(log_probabilities)) if log_probabilities else 0.0
        return Prediction(
            ids=ids,
            confidence=confidence,
            syntax_valid=not problems,
            tree=tree,
            tokens=tokens,
        )
