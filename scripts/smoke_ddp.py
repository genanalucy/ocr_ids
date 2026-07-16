#!/usr/bin/env python3
"""Small NCCL + model backward smoke test for the remote GPU host."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ocr_ids.models import VisionIDSModel


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = VisionIDSModel(
        32,
        pretrained=False,
        decoder_layers=1,
        image_size=224,
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    images = torch.zeros(1, 3, 224, 224, device=device)
    tokens = torch.ones(1, 5, dtype=torch.long, device=device)
    loss = model(images, tokens).float().mean()
    loss.backward()
    result = torch.tensor([rank + 1], device=device)
    dist.all_reduce(result)
    if rank == 0:
        expected = dist.get_world_size() * (dist.get_world_size() + 1) // 2
        assert result.item() == expected
        print(f"ddp_smoke=ok world_size={dist.get_world_size()} all_reduce={result.item():.0f}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

