import argparse
import dataclasses
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm

torch.set_float32_matmul_precision("high")

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - dependency guard for optional installs
    SentenceTransformer = None  # type: ignore[assignment]


@dataclasses.dataclass
class TrainConfig:
    """Hyperparameters and dataset choices for multimodal SlotFormer training."""

    dataset: str = "imagenette"  # choices: imagenette, coco
    data_root: str = "./datasets"
    coco_ann: str = "./datasets/coco/annotations/captions_train2017.json"
    image_encoder: str = "vit_base_patch14_dinov2"
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_size: int = 224
    num_slots: int = 80
    text_slots: int = 6
    transformer_layers: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    epochs: int = 40
    precision: str = "bf16-mixed"
    num_workers: int = 8
    log_interval: int = 25
    checkpoint_interval: int = 5
    recon_weight: float = 1.0
    text_weight: float = 0.5
    contrast_weight: float = 0.2
    contrast_temperature: float = 0.07
    compile_model: bool = False
    image_pretrained: bool = True
    max_steps_per_epoch: Optional[int] = None
    importance_weight: float = 0.0
    importance_margin: float = 0.0
    importance_prior_weight: float = 0.0
    importance_prior_decay: float = 0.1
    importance_temperature: float = 0.1
    importance_warmup_epochs: int = 0
    importance_prior_target: float = 1.0
    progressive_text_weight: float = 0.0
    progressive_max_prefix: int = 16
    progressive_decay: float = 0.5
    pca_warmstart_batches: int = 0
    curriculum_min_fraction: float = 1.0
    curriculum_warmup_epochs: int = 0


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class CocoCaptionDataset(Dataset[Tuple[Tensor, str]]):
    """MS-COCO (2017) captions with a single randomly sampled caption per image."""

    def __init__(self, root: str, ann_file: str, transform: transforms.Compose) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"COCO images not found at {root}. Download train2017 and point --data-root there."
            )
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(
                f"COCO annotations not found at {ann_file}. Please download captions_train2017.json."
            )

        self.dataset = torchvision.datasets.CocoCaptions(
            root=root,
            annFile=ann_file,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        image, captions = self.dataset[idx]
        caption = random.choice(captions)
        return image, caption


class ImagenetteCaptionDataset(Dataset[Tuple[Tensor, str]]):
    """Imagenette with textual labels used as weak captions for quick prototyping."""

    def __init__(self, root: str, transform: transforms.Compose) -> None:
        download = not os.path.isdir(os.path.join(root, "imagenette2"))
        self.dataset = torchvision.datasets.Imagenette(
            root,
            split="train",
            transform=transform,
            download=download,
        )
        self.classes = self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        image, label = self.dataset[idx]
        caption = self.classes[label]
        return image, caption


def collate_with_captions(batch: Iterable[Tuple[Tensor, str]]) -> Tuple[Tensor, List[str]]:
    images, captions = zip(*batch)
    return torch.stack(list(images)), list(captions)


def warmstart_slot_queries(
    model: "MultimodalSlotFormer",
    dataloader: torch.utils.data.DataLoader,
    num_batches: int,
    fabric: Optional[L.Fabric] = None,
) -> None:
    if num_batches <= 0:
        return

    device = next(model.image_encoder.parameters()).device
    samples: List[Tensor] = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            patch_tokens = model.image_encoder.forward_features(images)
            patch_tokens = patch_tokens[:, model.image_encoder.num_prefix_tokens :]
            samples.append(patch_tokens.reshape(-1, model.embed_dim))

    if not samples:
        return

    tokens = torch.cat(samples, dim=0).to(model.slot_queries.dtype)
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    try:
        _, _, vt = torch.linalg.svd(tokens, full_matrices=False)
    except RuntimeError:
        cov = tokens.T @ tokens / max(tokens.shape[0] - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        components = eigenvectors[:, -model.num_slots :].T
    else:
        components = vt[: model.num_slots]

    components = components.unsqueeze(0)
    model.slot_queries.data.copy_(components)
    if fabric is not None:
        fabric.print(
            f"Initialized slot queries using PCA warmstart from {num_batches} batches "
            f"({tokens.shape[0]} tokens)."
        )


class SentenceTransformerEncoder(nn.Module):
    """Wrapper that exposes sentence-transformer embeddings as a torch.nn.Module."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        if SentenceTransformer is None:  # pragma: no cover - dependency guard
            raise ImportError(
                "sentence-transformers is required for text conditioning. Install it via pip."
            )

        self.model = SentenceTransformer(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.output_dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def forward(self, captions: List[str], device: torch.device) -> Tensor:
        if len(captions) == 0:
            raise ValueError("Received an empty batch of captions.")
        normalized: List[str] = []
        for caption in captions:
            if isinstance(caption, str):
                normalized.append(caption)
            elif isinstance(caption, (list, tuple)) and caption:
                normalized.append(str(caption[0]))
            else:
                normalized.append(str(caption))

        embeddings = self.model.encode(
            normalized,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        )
        return embeddings.detach().clone()


class MultimodalSlotFormer(nn.Module):
    """SlotFormer that conditions slot generation on both visual tokens and text embeddings."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.num_slots = cfg.num_slots
        self.text_slots = cfg.text_slots
        self.recon_weight = cfg.recon_weight
        self.text_weight = cfg.text_weight
        self.contrast_weight = cfg.contrast_weight
        self.importance_weight = cfg.importance_weight
        self.importance_margin = cfg.importance_margin
        self.importance_prior_weight = cfg.importance_prior_weight
        self.importance_prior_decay = cfg.importance_prior_decay
        self.importance_temperature = cfg.importance_temperature
        self.importance_warmup_epochs = cfg.importance_warmup_epochs
        self.importance_prior_target = cfg.importance_prior_target
        self.progressive_text_weight = cfg.progressive_text_weight
        self.progressive_max_prefix = cfg.progressive_max_prefix
        self.progressive_decay = cfg.progressive_decay
        self.curriculum_min_fraction = cfg.curriculum_min_fraction
        self.curriculum_warmup_epochs = cfg.curriculum_warmup_epochs

        self.image_encoder = timm.create_model(
            cfg.image_encoder,
            pretrained=cfg.image_pretrained,
            img_size=cfg.image_size,
        ).eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.embed_dim = getattr(self.image_encoder, "embed_dim")
        self.num_patches = getattr(self.image_encoder.patch_embed, "num_patches")

        self.text_encoder = SentenceTransformerEncoder(cfg.text_encoder)
        self.text_proj = nn.Linear(self.text_encoder.output_dim, self.embed_dim)
        self.text_slot_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * self.text_slots),
        )

        self.slot_queries = nn.Parameter(torch.randn(1, self.num_slots, self.embed_dim))
        self.null_slots = nn.Parameter(torch.zeros(1, self.num_slots, self.embed_dim))
        self.modality_embed = nn.Parameter(torch.randn(2, self.embed_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        nn.init.normal_(self.null_slots, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)

        if getattr(self.image_encoder, "pos_embed", None) is not None:
            prefix = self.image_encoder.num_prefix_tokens
            pos_embed_init = self.image_encoder.pos_embed[:, prefix:].clone()
        else:  # dinov3 exposes ROPE, fall back to learned parameters
            pos_embed_init = torch.empty(1, self.num_patches, self.embed_dim)
            nn.init.normal_(pos_embed_init, std=0.02)
        self.decoder_pos_embed = nn.Parameter(pos_embed_init)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.generator = nn.TransformerDecoder(decoder_layer, num_layers=cfg.transformer_layers)
        self.reconstructor = nn.TransformerDecoder(decoder_layer, num_layers=cfg.transformer_layers)

        self.register_buffer(
            "tgt_mask",
            nn.Transformer.generate_square_subsequent_mask(self.num_slots),
            persistent=False,
        )

        self.text_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.contrast_temperature = nn.Parameter(
            torch.tensor(cfg.contrast_temperature, dtype=torch.float32)
        )
        self._current_epoch: Optional[int] = None
        self._total_epochs: Optional[int] = None

    def set_training_context(self, epoch: int, total_epochs: int) -> None:
        self._current_epoch = epoch
        self._total_epochs = max(total_epochs, 1)

    def _training_progress(self) -> float:
        if self._current_epoch is None or self._total_epochs is None:
            return 1.0
        if self._total_epochs <= 1:
            return 1.0
        return min(max(self._current_epoch / (self._total_epochs - 1), 0.0), 1.0)

    def _importance_scale(self) -> float:
        if self.importance_warmup_epochs <= 0 or self._current_epoch is None:
            return 1.0
        progress = (self._current_epoch + 1) / self.importance_warmup_epochs
        return min(max(progress, 0.0), 1.0)

    def _prior_scale(self) -> float:
        base = self._importance_scale()
        return min(base * self.importance_prior_target, 1.0)

    def _curriculum_slots(self) -> int:
        if self.curriculum_min_fraction >= 1.0:
            return self.num_slots
        if self.curriculum_warmup_epochs <= 0 or self._current_epoch is None:
            frac = 1.0
        else:
            frac = (self._current_epoch + 1) / self.curriculum_warmup_epochs
            frac = min(max(frac, 0.0), 1.0)
        min_slots = int(self.num_slots * self.curriculum_min_fraction)
        max_slots = self.num_slots
        allowed = min_slots + int((max_slots - min_slots) * frac)
        return max(min(max_slots, allowed), min_slots)

    def forward(
        self, images: Tensor, captions: Optional[List[str]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = images.device
        batch_size = images.shape[0]

        patch_tokens, memory, text_features = self._prepare_tokens(images, captions)
        slots = self._generate_slots(memory)

        # ensure at least text_slots slots remain active when conditioning on text
        min_prefix = 1 if captions is None else max(1, self.text_slots)
        curriculum_limit = self._curriculum_slots() if self.training else self.num_slots
        max_prefix = max(min(curriculum_limit, self.num_slots), min_prefix)
        prefix_bounds = (min_prefix, max_prefix + 1)
        keep_prefix = torch.randint(*prefix_bounds, (batch_size,), device=device)
        token_idx = torch.arange(self.num_slots, device=device)
        keep_mask = token_idx.unsqueeze(0) < keep_prefix.unsqueeze(1)

        null = self.null_slots.type_as(slots).expand(batch_size, -1, -1)
        masked_slots = torch.where(keep_mask.unsqueeze(-1), slots, null)

        reconstructed_patches = self._decode_slots(masked_slots)
        loss_recon = F.mse_loss(reconstructed_patches, patch_tokens)

        loss_text = torch.tensor(0.0, device=device)
        loss_progressive = torch.tensor(0.0, device=device)
        loss_contrast = torch.tensor(0.0, device=device)
        importance_loss = torch.tensor(0.0, device=device)
        importance_prior_loss = torch.tensor(0.0, device=device)
        slot_norms = slots.norm(dim=-1)
        if captions is not None:
            slot_summary = slots[:, : self.text_slots].mean(dim=1)
            predicted_text = self.text_head(slot_summary)
            loss_text = F.mse_loss(predicted_text, text_features)

            slot_norm = F.normalize(predicted_text, dim=-1)
            text_norm = F.normalize(text_features, dim=-1)
            temperature = self.contrast_temperature.clamp(min=1e-3)
            logits = slot_norm @ text_norm.t() / temperature
            labels = torch.arange(batch_size, device=device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
            loss_contrast = 0.5 * (loss_i2t + loss_t2i)

            if self.progressive_text_weight > 0:
                max_prefix = min(self.num_slots, self.progressive_max_prefix)
                decay_factors = torch.exp(
                    -self.progressive_decay
                    * torch.arange(max_prefix, device=device, dtype=slots.dtype)
                )
                for k, weight in zip(range(1, max_prefix + 1), decay_factors):
                    prefix_summary = slots[:, :k].mean(dim=1)
                    prefix_prediction = self.text_head(prefix_summary)
                    mse = F.mse_loss(prefix_prediction, text_features)
                    loss_progressive = loss_progressive + weight * mse

        if self.importance_weight > 0:
            diffs = slot_norms[:, 1:] - slot_norms[:, :-1] + self.importance_margin
            importance_loss = F.relu(diffs).mean()

        if self.importance_prior_weight > 0:
            mean_norm = slot_norms.mean(dim=0)
            temperature = max(self.importance_temperature, 1e-6)
            norm_logits = mean_norm / temperature
            prob = torch.softmax(norm_logits, dim=0)
            prior_idx = torch.arange(self.num_slots, device=device, dtype=prob.dtype)
            prior_logits = -self.importance_prior_decay * prior_idx
            prior = torch.softmax(prior_logits, dim=0)
            importance_prior_loss = torch.sum(
                prob * (torch.log(prob + 1e-8) - torch.log(prior + 1e-8))
            )

        importance_scale = self._importance_scale()
        prior_scale = self._prior_scale()

        total_loss = (
            self.recon_weight * loss_recon
            + self.text_weight * loss_text
            + self.contrast_weight * loss_contrast
            + self.progressive_text_weight * loss_progressive
            + (self.importance_weight * importance_scale) * importance_loss
            + (self.importance_prior_weight * prior_scale) * importance_prior_loss
        )

        logs = {
            "loss": total_loss.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_text": loss_text.detach(),
            "loss_contrast": loss_contrast.detach(),
            "loss_progressive": loss_progressive.detach(),
            "loss_importance": importance_loss.detach(),
            "loss_prior": importance_prior_loss.detach(),
            "importance_scale": torch.tensor(importance_scale, device=device),
            "prior_scale": torch.tensor(prior_scale, device=device),
            "max_slots": torch.tensor(max_prefix, device=device, dtype=torch.float32),
        }
        return total_loss, logs

    def reconstruct_with_budget(
        self,
        images: Tensor,
        captions: Optional[List[str]],
        slot_keep: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Reconstruct image patches using a deterministic slot keep mask.

        Args:
            images: normalized input images matching training preprocessing.
            captions: optional list of captions for conditioning.
            slot_keep: boolean tensor of shape [B, num_slots] where True keeps a slot.

        Returns:
            reconstructed_patches, original_patch_tokens, slots, text_features
        """

        patch_tokens, memory, text_features = self._prepare_tokens(images, captions)
        slots = self._generate_slots(memory)

        if slot_keep.dtype != torch.bool:
            slot_keep = slot_keep.bool()
        keep_mask = slot_keep.to(slots.device)
        null = self.null_slots.type_as(slots).expand_as(slots)
        masked_slots = torch.where(keep_mask.unsqueeze(-1), slots, null)
        reconstructed = self._decode_slots(masked_slots)
        return reconstructed, patch_tokens, slots, text_features

    def _prepare_tokens(
        self, images: Tensor, captions: Optional[List[str]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        device = images.device
        batch_size = images.shape[0]

        with torch.no_grad():
            patch_tokens = self.image_encoder.forward_features(images)
            patch_tokens = patch_tokens[:, self.image_encoder.num_prefix_tokens :]
        patch_tokens = patch_tokens + self.modality_embed[0]

        text_tokens: Optional[Tensor] = None
        text_features = torch.zeros(batch_size, self.embed_dim, device=device)
        if captions is not None:
            text_embeddings = self.text_encoder(captions, device=device)
            text_features = self.text_proj(text_embeddings)
            text_tokens = self.text_slot_proj(text_features).view(
                batch_size, self.text_slots, self.embed_dim
            )
            text_tokens = text_tokens + self.modality_embed[1]

        if text_tokens is not None:
            memory = torch.cat([text_tokens, patch_tokens], dim=1)
        else:
            memory = patch_tokens

        return patch_tokens, memory, text_features

    def _generate_slots(self, memory: Tensor) -> Tensor:
        batch_size = memory.shape[0]
        device = memory.device
        return self.generator(
            tgt=self.slot_queries.expand(batch_size, -1, -1),
            memory=memory,
            tgt_mask=self.tgt_mask.to(device),
            tgt_is_causal=True,
        )

    def _decode_slots(self, slots: Tensor) -> Tensor:
        batch_size = slots.shape[0]
        return self.reconstructor(
            tgt=self.decoder_pos_embed.expand(batch_size, -1, -1),
            memory=slots,
        )


def build_dataloader(cfg: TrainConfig) -> DataLoader[Tuple[Tensor, List[str]]]:
    transform = build_transform(cfg.image_size)

    if cfg.dataset == "coco":
        image_root = os.path.join(cfg.data_root, "coco", "train2017")
        dataset = CocoCaptionDataset(image_root, cfg.coco_ann, transform)
    elif cfg.dataset == "imagenette":
        dataset = ImagenetteCaptionDataset(cfg.data_root, transform)
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset}")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_with_captions,
    )


def parse_config() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train multimodal SlotFormer")
    parser.add_argument("--dataset", choices=["imagenette", "coco"], default="imagenette")
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--coco-ann", default="./datasets/coco/annotations/captions_train2017.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-slots", type=int)
    parser.add_argument("--text-slots", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--importance-weight", type=float)
    parser.add_argument("--importance-margin", type=float)
    parser.add_argument("--importance-prior-weight", type=float)
    parser.add_argument("--importance-prior-decay", type=float)
    parser.add_argument("--importance-temperature", type=float)
    parser.add_argument("--importance-warmup-epochs", type=int)
    parser.add_argument("--importance-prior-target", type=float)
    parser.add_argument("--progressive-text-weight", type=float)
    parser.add_argument("--progressive-max-prefix", type=int)
    parser.add_argument("--progressive-decay", type=float)
    parser.add_argument("--pca-warmstart-batches", type=int)
    parser.add_argument("--curriculum-min-fraction", type=float)
    parser.add_argument("--curriculum-warmup-epochs", type=int)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.dataset = args.dataset
    cfg.data_root = args.data_root
    cfg.coco_ann = args.coco_ann
    cfg.precision = args.precision
    cfg.compile_model = bool(args.compile)
    cfg.image_pretrained = not args.no_pretrained

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_slots is not None:
        cfg.num_slots = args.num_slots
    if args.text_slots is not None:
        cfg.text_slots = args.text_slots
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.max_steps is not None:
        cfg.max_steps_per_epoch = max(args.max_steps, 1)
    if args.checkpoint_interval is not None:
        cfg.checkpoint_interval = max(args.checkpoint_interval, 1)
    if args.importance_weight is not None:
        cfg.importance_weight = max(args.importance_weight, 0.0)
    if args.importance_margin is not None:
        cfg.importance_margin = args.importance_margin
    if args.importance_prior_weight is not None:
        cfg.importance_prior_weight = max(args.importance_prior_weight, 0.0)
    if args.importance_prior_decay is not None:
        cfg.importance_prior_decay = max(args.importance_prior_decay, 0.0)
    if args.importance_temperature is not None:
        cfg.importance_temperature = max(args.importance_temperature, 1e-6)
    if args.importance_warmup_epochs is not None:
        cfg.importance_warmup_epochs = max(args.importance_warmup_epochs, 0)
    if args.importance_prior_target is not None:
        cfg.importance_prior_target = max(args.importance_prior_target, 0.0)
    if args.progressive_text_weight is not None:
        cfg.progressive_text_weight = max(args.progressive_text_weight, 0.0)
    if args.progressive_max_prefix is not None:
        cfg.progressive_max_prefix = max(args.progressive_max_prefix, 1)
    if args.progressive_decay is not None:
        cfg.progressive_decay = max(args.progressive_decay, 0.0)
    if args.pca_warmstart_batches is not None:
        cfg.pca_warmstart_batches = max(args.pca_warmstart_batches, 0)
    if args.curriculum_min_fraction is not None:
        cfg.curriculum_min_fraction = min(max(args.curriculum_min_fraction, 0.0), 1.0)
    if args.curriculum_warmup_epochs is not None:
        cfg.curriculum_warmup_epochs = max(args.curriculum_warmup_epochs, 0)

    return cfg


def train() -> None:
    cfg = parse_config()

    fabric = L.Fabric(accelerator="auto", precision=cfg.precision)
    fabric.launch()

    dataloader = build_dataloader(cfg)

    model = MultimodalSlotFormer(cfg)

    if cfg.pca_warmstart_batches > 0:
        warmstart_slot_queries(model, dataloader, cfg.pca_warmstart_batches, fabric)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    if cfg.compile_model:
        model = torch.compile(model, fullgraph=False, disable=False)  # type: ignore[arg-type]

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    checkpoint_dir = os.path.join("checkpoints", "slotformer")
    if fabric.global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.set_training_context(epoch, cfg.epochs)
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch:02d}",
            leave=False,
        )

        for step, (images, captions) in progress:
            images = images.to(fabric.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = model(images, captions)
            fabric.backward(loss)
            optimizer.step()

            if (step + 1) % cfg.log_interval == 0:
                postfix = {k: f"{v.item():.4f}" for k, v in logs.items()}
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                progress.set_postfix(postfix)

            if cfg.max_steps_per_epoch is not None and (step + 1) >= cfg.max_steps_per_epoch:
                break

        if (epoch + 1) % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:02d}.ckpt")
            fabric.save(ckpt_path, {"model": model, "optimizer": optimizer, "config": dataclasses.asdict(cfg)})

    fabric.print("Training complete.")
    final_path = os.path.join(checkpoint_dir, "final.ckpt")
    fabric.save(final_path, {"model": model, "optimizer": optimizer, "config": dataclasses.asdict(cfg)})


if __name__ == "__main__":
    train()
