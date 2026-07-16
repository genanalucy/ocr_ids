#!/usr/bin/env python3
"""Minimal reproducible training entrypoint for the phase-one baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

import torch
from PIL import Image
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
import yaml

from ocr_ids.dataset import SampleRecord, read_jsonl
from ocr_ids.models import VisionIDSModel
from ocr_ids.vocab import Vocabulary, build_vocabulary


class GlyphDataset(Dataset):
    def __init__(self, records: list[SampleRecord], vocabulary: Vocabulary, image_size: int) -> None:
        self.records = [record for record in records if record.image_path]
        self.vocabulary = vocabulary
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=2, translate=(0.03, 0.03), scale=(0.92, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[int]]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, self.vocabulary.encode(record.ids)


def collate(batch: list[tuple[torch.Tensor, list[int]]], pad_id: int):
    images = torch.stack([item[0] for item in batch])
    max_length = max(len(item[1]) for item in batch)
    sequences = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for row, (_, sequence) in enumerate(batch):
        sequences[row, : len(sequence)] = torch.tensor(sequence)
    return images, sequences


def expand_environment(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def distributed_context() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return distributed, rank, local_rank, world_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = expand_environment(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    distributed, rank, local_rank, world_size = distributed_context()
    seed = int(config.get("seed", 42)) + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    records = list(read_jsonl(config["data"]["train_manifest"]))
    output = Path(config["output_dir"])
    vocabulary = build_vocabulary(record.ids for record in records)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        vocabulary.save(output / "vocab.json")
    token_ids = vocabulary.token_to_id

    dataset = GlyphDataset(records, vocabulary, config["data"]["image_size"])
    if not dataset:
        raise RuntimeError("训练 manifest 中没有可用 image_path")
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["training"]["num_workers"] > 0,
        collate_fn=lambda batch: collate(batch, token_ids["<pad>"]),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    model = VisionIDSModel(
        len(vocabulary.tokens),
        encoder_name=config["model"]["encoder_name"],
        pretrained=config["model"]["pretrained"],
        encoder_checkpoint=config["model"].get("encoder_checkpoint"),
        decoder_dim=config["model"]["decoder_dim"],
        decoder_layers=config["model"]["decoder_layers"],
        decoder_heads=config["model"]["decoder_heads"],
        max_length=config["model"]["max_length"],
        image_size=config["data"]["image_size"],
    ).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_function = nn.CrossEntropyLoss(ignore_index=token_ids["<pad>"])
    precision = config["training"].get("precision", "bf16")
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))

    model.train()
    global_step = 0
    for epoch in range(config["training"]["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, sequences) in enumerate(loader):
            images, sequences = images.to(device), sequences.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images, sequences[:, :-1])
                loss = loss_function(
                    logits.reshape(-1, logits.shape[-1]), sequences[:, 1:].reshape(-1)
                )
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if rank == 0 and global_step % config["training"]["log_every"] == 0:
                effective_batch = (
                    config["training"]["batch_size"] * world_size * accumulation_steps
                )
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={loss.item() * accumulation_steps:.4f} global_batch={effective_batch}"
                )
        if rank == 0:
            state_model = model.module if isinstance(model, DistributedDataParallel) else model
            torch.save(
                {"model": state_model.state_dict(), "epoch": epoch + 1, "config": config},
                output / "last.pt",
            )
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
