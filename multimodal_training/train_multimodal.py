"""
Training script for Slot-CoCa with Distributed Data Parallel (DDP)
Supports multi-GPU training on 2x A100
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import os
import json
from tqdm import tqdm

from models.slot_coca import SlotCoCa
try:
    from imagenette_captions_dataset import ImagenetteWithCaptions, collate_fn
    IMAGENETTE_AVAILABLE = True
except:
    IMAGENETTE_AVAILABLE = False

try:
    from imagenet_captions_dataset import ImageNetCaptionsDataset, collate_fn as collate_fn_imagenet
    IMAGENET_CAPTIONS_AVAILABLE = True
except:
    IMAGENET_CAPTIONS_AVAILABLE = False

try:
    from coco_dataset import COCOCaptionsDataset, collate_fn as collate_fn_coco
    COCO_AVAILABLE = True
except:
    COCO_AVAILABLE = False


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


class MultimodalTrainer:
    def __init__(self, args, rank, world_size):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Create model
        self.model = SlotCoCa(
            num_slots=args.num_slots,
            num_layers=args.num_layers,
            encoder_name=args.encoder_name,
            projection_dim=args.projection_dim,
            temperature=args.temperature
        ).to(self.device)
        
        # Wrap with DDP
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs
        )
        
        # Loss weights
        self.lambda_recon = args.lambda_recon
        self.lambda_contrast = args.lambda_contrast
        self.lambda_caption = args.lambda_caption
        
        # Logging
        if rank == 0:
            self.writer = SummaryWriter(log_dir=args.log_dir)
            self.checkpoint_dir = Path(args.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
    
    def prepare_data(self):
        """Prepare dataloaders with DDP"""
        # Detect dataset type
        use_coco = self.args.use_coco if hasattr(self.args, 'use_coco') else False
        use_imagenette = (Path(self.args.data_dir) / 'train').exists() and IMAGENETTE_AVAILABLE and not use_coco
        
        if use_coco:
            if not COCO_AVAILABLE:
                raise RuntimeError("COCO dataset not available! Install: pip install pycocotools")
            print(f"📁 Using COCO dataset from {self.args.data_dir}")
            train_dataset = COCOCaptionsDataset(
                root_dir=Path(self.args.data_dir) / 'train2017',
                ann_file=Path(self.args.data_dir) / 'annotations' / 'captions_train2017.json',
                captions_per_image=5
            )
            current_collate_fn = collate_fn_coco
        elif use_imagenette:
            # Use Imagenette dataset
            print(f"📁 Using Imagenette dataset from {self.args.data_dir}")
            train_dataset = ImagenetteWithCaptions(
                imagenette_root=self.args.data_dir,
                split='train',
                captions_per_image=5
            )
            current_collate_fn = collate_fn
        elif IMAGENET_CAPTIONS_AVAILABLE:
            # Use ImageNet-Captions dataset
            print(f"📁 Using ImageNet-Captions from {self.args.data_dir}")
            train_dataset = ImageNetCaptionsDataset(
                root_dir=self.args.data_dir,
                split='train',
                subset_size=self.args.subset_size
            )
            current_collate_fn = collate_fn_imagenet
        else:
            raise RuntimeError("No dataset available! Use --use_coco or install imagenette_captions_dataset.py")
        
        # Apply num_samples limit if specified
        if hasattr(self.args, 'num_samples') and self.args.num_samples:
            print(f"⚠️  Limiting to {self.args.num_samples} samples (from {len(train_dataset)})")
            train_dataset = torch.utils.data.Subset(train_dataset, range(min(self.args.num_samples, len(train_dataset))))
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            collate_fn=current_collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # Validation dataset
        if self.rank == 0:
            if use_coco:
                val_dataset = COCOCaptionsDataset(
                    root_dir=Path(self.args.data_dir) / 'val2017',
                    ann_file=Path(self.args.data_dir) / 'annotations' / 'captions_val2017.json',
                    captions_per_image=5
                )
            elif use_imagenette:
                val_dataset = ImagenetteWithCaptions(
                    imagenette_root=self.args.data_dir,
                    split='val',
                    captions_per_image=5
                )
            else:
                val_dataset = ImageNetCaptionsDataset(
                    root_dir=self.args.data_dir,
                    split='val',
                    subset_size=self.args.val_subset_size
                )
            
            # Limit val samples if specified
            if hasattr(self.args, 'num_samples') and self.args.num_samples:
                val_limit = min(1000, len(val_dataset))  # Max 1000 for validation
                val_dataset = torch.utils.data.Subset(val_dataset, range(val_limit))
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=current_collate_fn,
                num_workers=self.args.num_workers
            )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        losses_dict = {'recon': 0, 'contrast': 0, 'caption': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=(self.rank != 0))
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            captions = batch['captions']
            
            # Tokenize captions
            tokenizer = self.model.module.tokenizer
            text_tokens = tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt'
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(
                images=images,
                text_tokens=text_tokens,
                caption_tokens=text_tokens,
                mode='all'
            )
            
            # Compute weighted loss
            loss = (
                self.lambda_recon * outputs.get('reconstruction_loss', 0) +
                self.lambda_contrast * outputs.get('contrastive_loss', 0) +
                self.lambda_caption * outputs.get('caption_loss', 0)
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            if 'reconstruction_loss' in outputs:
                losses_dict['recon'] += outputs['reconstruction_loss'].item()
            if 'contrastive_loss' in outputs:
                losses_dict['contrast'] += outputs['contrastive_loss'].item()
            if 'caption_loss' in outputs:
                losses_dict['caption'] += outputs['caption_loss'].item()
            
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon': outputs.get('reconstruction_loss', 0).item() if 'reconstruction_loss' in outputs else 0,
                    'contrast': outputs.get('contrastive_loss', 0).item() if 'contrastive_loss' in outputs else 0
                })
                
                if self.global_step % self.args.log_interval == 0:
                    self.writer.add_scalar('train/total_loss', loss.item(), self.global_step)
                    for k, v in outputs.items():
                        if 'loss' in k:
                            self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        for k in losses_dict:
            losses_dict[k] /= len(self.train_loader)
        
        return avg_loss, losses_dict
    
    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        if self.rank != 0:
            return {}
        
        self.model.eval()
        total_loss = 0
        losses_dict = {'recon': 0, 'contrast': 0, 'caption': 0}
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['images'].to(self.device)
            captions = batch['captions']
            
            tokenizer = self.model.module.tokenizer
            text_tokens = tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(
                images=images,
                text_tokens=text_tokens,
                caption_tokens=text_tokens,
                mode='all'
            )
            
            loss = (
                self.lambda_recon * outputs.get('reconstruction_loss', 0) +
                self.lambda_contrast * outputs.get('contrastive_loss', 0) +
                self.lambda_caption * outputs.get('caption_loss', 0)
            )
            
            total_loss += loss.item()
            if 'reconstruction_loss' in outputs:
                losses_dict['recon'] += outputs['reconstruction_loss'].item()
            if 'contrastive_loss' in outputs:
                losses_dict['contrast'] += outputs['contrastive_loss'].item()
            if 'caption_loss' in outputs:
                losses_dict['caption'] += outputs['caption_loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        for k in losses_dict:
            losses_dict[k] /= len(self.val_loader)
        
        return avg_loss, losses_dict
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'args': vars(self.args)
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"✅ Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"🏆 Saved best model: {best_path}")
    
    def train(self):
        """Main training loop"""
        self.prepare_data()
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            if self.rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{self.args.epochs}")
                print(f"{'='*60}")
            
            # Train
            train_loss, train_losses = self.train_epoch(epoch)
            
            # Validate
            if self.rank == 0:
                val_loss, val_losses = self.validate()
                
                # Log to tensorboard
                self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
                self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
                self.writer.add_scalar('epoch/lr', self.scheduler.get_last_lr()[0], epoch)
                
                print(f"\nTrain Loss: {train_loss:.4f}")
                print(f"  - Reconstruction: {train_losses['recon']:.4f}")
                print(f"  - Contrastive: {train_losses['contrast']:.4f}")
                print(f"  - Caption: {train_losses['caption']:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                
                # Save checkpoint
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                if (epoch + 1) % self.args.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
            
            # Step scheduler
            self.scheduler.step()
            
            # Synchronize
            if self.world_size > 1:
                dist.barrier()


def main_worker(rank, world_size, args):
    """Worker function for each GPU"""
    setup_ddp(rank, world_size)
    
    trainer = MultimodalTrainer(args, rank, world_size)
    trainer.train()
    
    cleanup_ddp()


def single_device_training(args):
    """Simple single-device training (no DDP)"""
    device = torch.device(args.device)
    
    # Create model
    model = SlotCoCa(
        num_slots=args.num_slots,
        num_layers=args.num_layers,
        encoder_name=args.encoder_name,
        projection_dim=args.projection_dim,
        temperature=args.temperature
    ).to(device)
    
    # Prepare data
    use_coco = args.use_coco if hasattr(args, 'use_coco') else False
    
    if use_coco:
        if not COCO_AVAILABLE:
            raise RuntimeError("COCO not available! Install: pip install pycocotools")
        print(f"📁 Using COCO from {args.data_dir}")
        train_dataset = COCOCaptionsDataset(
            root_dir=Path(args.data_dir) / 'train2017',
            ann_file=Path(args.data_dir) / 'annotations' / 'captions_train2017.json',
            captions_per_image=5
        )
        current_collate_fn = collate_fn_coco
    else:
        print(f"📁 Using Imagenette from {args.data_dir}")
        train_dataset = ImagenetteWithCaptions(
            imagenette_root=args.data_dir,
            split='train',
            captions_per_image=5
        )
        current_collate_fn = collate_fn
    
    # Limit samples if specified
    if args.num_samples:
        print(f"⚠️  Limiting to {args.num_samples} samples")
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.num_samples, len(train_dataset))))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=current_collate_fn,
        num_workers=args.num_workers
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['images'].to(device)
            captions = batch['captions']
            
            # Tokenize captions
            caption_tokens = model.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            outputs = model(
                images=images,
                text_tokens=caption_tokens,
                caption_tokens=caption_tokens,
                mode='all'
            )
            
            # Combined loss
            loss = (
                args.lambda_recon * outputs.get('recon_loss', 0) +
                args.lambda_contrast * outputs.get('contrast_loss', 0) +
                args.lambda_caption * outputs.get('caption_loss', 0)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{outputs.get('recon_loss', 0):.4f}",
                'contrast': f"{outputs.get('contrast_loss', 0):.4f}",
                'caption': f"{outputs.get('caption_loss', 0):.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = Path(args.checkpoint_dir) / "best_model.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, final_path)
    print(f"✅ Training complete! Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Slot-CoCa')
    
    # Model args
    parser.add_argument('--num_slots', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--encoder_name', type=str, default='vit_base_patch16_dinov3')
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    
    # Data args
    parser.add_argument('--data_dir', type=str, default='./data/imagenet_captions')
    parser.add_argument('--use_coco', action='store_true', help='Use COCO Captions dataset')
    parser.add_argument('--num_samples', type=int, default=None, help='Limit number of training samples')
    parser.add_argument('--subset_size', type=int, default=100000, help='Use subset for proof-of-concept')
    parser.add_argument('--val_subset_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_contrast', type=float, default=1.0)
    parser.add_argument('--lambda_caption', type=float, default=1.0)
    
    # Logging args
    parser.add_argument('--log_dir', type=str, default='./logs/slot_coca')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/slot_coca')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    
    # Device args
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--single_device', action='store_true', help='Use single device (no DDP)')
    
    # DDP args
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs for DDP')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SLOT-COCA MULTIMODAL TRAINING")
    print("="*60)
    
    # Single device training (CPU or single GPU)
    if args.single_device or args.device == 'cpu':
        print(f"Device: {args.device}")
        print(f"Batch size: {args.batch_size}")
        if args.num_samples:
            print(f"Samples: {args.num_samples}")
        print(f"Epochs: {args.epochs}")
        print("="*60)
        
        # Run single-device training
        single_device_training(args)
    else:
        # DDP training
        print(f"GPUs: {args.world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * args.world_size}")
        if args.num_samples:
            print(f"Samples: {args.num_samples}")
        print(f"Epochs: {args.epochs}")
        print("="*60)
        
        # Launch DDP
        torch.multiprocessing.spawn(
            main_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )


if __name__ == '__main__':
    main()
