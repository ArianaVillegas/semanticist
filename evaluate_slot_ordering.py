import argparse
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import (
    CocoCaptionDataset,
    ImagenetteCaptionDataset,
    MultimodalSlotFormer,
    TrainConfig,
    collate_with_captions,
    build_transform,
)


@torch.no_grad()
def compute_metrics(
    model: MultimodalSlotFormer,
    dataloader: DataLoader,
    slot_budgets: List[int],
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    device = next(model.parameters()).device
    num_slots = model.num_slots
    slot_indices = torch.arange(num_slots, device=device)

    mse_stats: Dict[int, Dict[str, float]] = {
        k: {"mse_sum": 0.0, "token_count": 0.0} for k in slot_budgets
    }
    retrieval_preds: Dict[int, List[torch.Tensor]] = {k: [] for k in slot_budgets}
    retrieval_targets: List[torch.Tensor] = []
    slot_energy = torch.zeros(num_slots, device=device)
    sample_count = 0

    for batch_idx, (images, captions) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        patch_tokens, memory, text_features = model._prepare_tokens(images, captions)
        slots = model._generate_slots(memory)
        slot_energy += slots.pow(2).sum(dim=(0, 2))
        reconstruction_null = model.null_slots.type_as(slots).expand_as(slots)

        for k in slot_budgets:
            keep_mask = slot_indices.unsqueeze(0) < k
            masked_slots = torch.where(keep_mask.unsqueeze(-1), slots, reconstruction_null)
            recon = model._decode_slots(masked_slots)
            mse = F.mse_loss(recon, patch_tokens, reduction="sum")
            mse_stats[k]["mse_sum"] += mse.item()
            mse_stats[k]["token_count"] += patch_tokens.numel()

            slot_summary = slots[:, :k].mean(dim=1)
            predicted_text = model.text_head(slot_summary)
            retrieval_preds[k].append(F.normalize(predicted_text.detach().cpu(), dim=-1))

        retrieval_targets.append(F.normalize(text_features.detach().cpu(), dim=-1))
        sample_count += images.size(0)

    metrics: Dict[str, Dict[str, float]] = {}
    total_energy = slot_energy.sum().item() + 1e-8

    for k in slot_budgets:
        mse_sum = mse_stats[k]["mse_sum"]
        token_count = max(mse_stats[k]["token_count"], 1.0)
        metrics.setdefault(str(k), {})["token_mse"] = mse_sum / token_count

        preds = torch.cat(retrieval_preds[k], dim=0)
        targets = torch.cat(retrieval_targets, dim=0)
        sims = preds @ targets.T
        recall1 = (sims.argmax(dim=1) == torch.arange(sample_count)).float().mean().item()
        metrics[str(k)]["recall@1"] = recall1

        energy_ratio = slot_energy[:k].sum().item() / total_energy
        metrics[str(k)]["energy_ratio"] = energy_ratio

    metrics["meta"] = {
        "samples": sample_count,
        "slot_energy": slot_energy.cpu().tolist(),
    }
    return metrics


def build_eval_dataloader(cfg: TrainConfig, dataset_name: str, batch_size: int, max_batches: Optional[int]) -> DataLoader:
    transform = build_transform(cfg.image_size)

    if dataset_name == "coco":
        image_root = os.path.join(cfg.data_root, "coco", "train2017")
        dataset = CocoCaptionDataset(image_root, cfg.coco_ann, transform)
    elif dataset_name == "imagenette":
        dataset = ImagenetteCaptionDataset(cfg.data_root, transform)
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    effective_batch = min(batch_size, len(dataset))
    return DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_with_captions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate slot ordering metrics")
    parser.add_argument("--checkpoint", required=True, help="Path to Fabric checkpoint")
    parser.add_argument("--dataset", choices=["coco", "imagenette"], default="coco")
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument(
        "--slots",
        type=str,
        default="4,8,16,32,48,64",
        help="Comma separated list of slot budgets to evaluate",
    )
    parser.add_argument("--output", default="slot_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slot_budgets = sorted({int(s) for s in args.slots.split(",") if s.strip()})

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = checkpoint.get("config", {})
    cfg = TrainConfig(**cfg_dict)
    cfg.data_root = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSlotFormer(cfg)
    model.load_state_dict(checkpoint["model"])
    model.set_training_context(cfg.epochs - 1, cfg.epochs)
    model.to(device)
    model.eval()

    dataloader = build_eval_dataloader(cfg, args.dataset, args.batch_size, args.max_batches)
    metrics = compute_metrics(model, dataloader, slot_budgets, max_batches=args.max_batches)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
