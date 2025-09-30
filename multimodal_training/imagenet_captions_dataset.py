"""
ImageNet-Captions Dataset Loader

Downloads and prepares the ImageNet-Captions dataset for training.
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from pathlib import Path
from tqdm import tqdm
import requests

class ImageNetCaptionsDataset(Dataset):
    """
    ImageNet with captions dataset
    
    Uses ImageNet images with 5 captions per image from various sources:
    - ImageNet-Captions paper
    - Or CLIP-generated captions as fallback
    """
    def __init__(
        self,
        root_dir,
        split='train',
        transform=None,
        max_caption_length=77,
        subset_size=None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or self._default_transform()
        self.max_caption_length = max_caption_length
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        if subset_size:
            self.annotations = self.annotations[:subset_size]
        
        print(f"Loaded {len(self.annotations)} image-caption pairs for {split}")
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_annotations(self):
        """Load image-caption annotations"""
        annotation_file = self.root_dir / f'{self.split}_captions.json'
        
        if not annotation_file.exists():
            print(f"⚠️  Annotation file not found: {annotation_file}")
            print("Creating synthetic captions from ImageNet class names...")
            return self._create_synthetic_captions()
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        return data['annotations']
    
    def _create_synthetic_captions(self):
        """
        Create synthetic captions from ImageNet class names
        Useful for initial testing before downloading full dataset
        """
        from torchvision.datasets import ImageNet
        
        # Try to load ImageNet
        try:
            imagenet = ImageNet(root=str(self.root_dir / 'imagenet'), split=self.split)
        except:
            print("❌ ImageNet not found. Please download ImageNet first.")
            return []
        
        annotations = []
        for idx, (_, class_idx) in enumerate(imagenet):
            class_name = imagenet.classes[class_idx][0]
            # Generate 5 template captions per image
            templates = [
                f"A photo of a {class_name}",
                f"An image showing a {class_name}",
                f"This is a {class_name}",
                f"A {class_name} in the image",
                f"Picture of a {class_name}"
            ]
            
            for caption in templates:
                annotations.append({
                    'image_id': idx,
                    'image_path': imagenet.imgs[idx][0],
                    'caption': caption
                })
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        image_path = ann['image_path']
        if not Path(image_path).is_absolute():
            image_path = self.root_dir / image_path
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = torch.zeros(3, 224, 224)
        
        caption = ann['caption']
        
        return {
            'image': image,
            'caption': caption,
            'image_id': ann.get('image_id', idx)
        }


def download_imagenet_captions(root_dir, subset='100k'):
    """
    Download ImageNet-Captions dataset
    
    Args:
        root_dir: Directory to save the dataset
        subset: '100k' for proof-of-concept or 'full' for complete dataset
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading ImageNet-Captions ({subset} subset)...")
    
    if subset == '100k':
        # For proof-of-concept, use COCO Captions or a subset
        print("⚠️  For proof-of-concept, using COCO Captions style format")
        print("Please download ImageNet images separately")
        print("\nYou can download ImageNet from:")
        print("  https://www.image-net.org/download.php")
        print("\nFor captions, we'll generate synthetic ones from class names")
        print("Or you can manually download from:")
        print("  https://github.com/mlfoundations/imagenet-captions")
    else:
        print("Downloading full ImageNet-Captions...")
        # Add actual download URLs when available
        print("Please download manually from:")
        print("  https://github.com/mlfoundations/imagenet-captions")
    
    return root_dir


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'captions': captions,
        'image_ids': image_ids
    }


if __name__ == '__main__':
    # Test dataset loading
    print("Testing ImageNet-Captions dataset...")
    
    # Download dataset
    root_dir = download_imagenet_captions('./data/imagenet_captions', subset='100k')
    
    # Create dataset
    dataset = ImageNetCaptionsDataset(
        root_dir=root_dir,
        split='train',
        subset_size=100
    )
    
    print(f"\n✅ Dataset loaded: {len(dataset)} samples")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Caption: {sample['caption']}")
        print(f"  Image ID: {sample['image_id']}")
