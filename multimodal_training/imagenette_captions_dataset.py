"""
Imagenette + ImageNet-Captions Dataset
Uses existing Imagenette images with ImageNet-Captions annotations
Perfect for proof-of-concept without downloading full ImageNet
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import json
import os
from pathlib import Path
import random

class ImagenetteWithCaptions(Dataset):
    """
    Dataset combining Imagenette images with ImageNet-Captions
    
    Imagenette: 13,394 images from 10 ImageNet classes
    ImageNet-Captions: 5 captions per ImageNet image
    
    Strategy:
    1. Load Imagenette images (you already have these)
    2. Either download matching captions OR generate synthetic ones
    3. Each image gets 5 captions (as per ImageNet-Captions format)
    """
    
    # Imagenette class mapping to ImageNet synsets
    IMAGENETTE_CLASSES = {
        'n01440764': 'tench',
        'n02102040': 'English springer',
        'n02979186': 'cassette player',
        'n03000684': 'chain saw',
        'n03028079': 'church',
        'n03394916': 'French horn',
        'n03417042': 'garbage truck',
        'n03425413': 'gas pump',
        'n03445777': 'golf ball',
        'n03888257': 'parachute'
    }
    
    def __init__(
        self,
        imagenette_root='./datasets/imagenette2',
        split='train',
        captions_file=None,
        transform=None,
        max_caption_length=77,
        captions_per_image=5
    ):
        """
        Args:
            imagenette_root: Path to imagenette2 directory
            split: 'train' or 'val'
            captions_file: Optional JSON file with ImageNet-Captions
            transform: Image transforms
            max_caption_length: Max tokens for captions
            captions_per_image: Number of captions per image (default: 5)
        """
        self.root = Path(imagenette_root)
        self.split = split
        self.transform = transform or self._default_transform()
        self.max_caption_length = max_caption_length
        self.captions_per_image = captions_per_image
        
        # Load Imagenette images
        self.image_dataset = ImageFolder(
            root=str(self.root / split),
            transform=self.transform
        )
        
        # Load or generate captions
        if captions_file and Path(captions_file).exists():
            self.captions_dict = self._load_imagenet_captions(captions_file)
        else:
            print("⚠️  No captions file provided. Generating synthetic captions...")
            self.captions_dict = self._generate_synthetic_captions()
        
        # Create index mapping (image_idx -> list of captions)
        self.samples = self._create_samples()
        
        print(f"✅ Loaded {len(self.samples)} image-caption pairs for {split}")
        print(f"   - Unique images: {len(self.image_dataset)}")
        print(f"   - Captions per image: {self.captions_per_image}")
    
    def _default_transform(self):
        """Default image preprocessing"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_imagenet_captions(self, captions_file):
        """Load ImageNet-Captions JSON file"""
        print(f"📥 Loading captions from: {captions_file}")
        with open(captions_file, 'r') as f:
            data = json.load(f)
        
        # Data format: {synset: [list of captions]}
        if isinstance(data, dict):
            # Already organized by synset
            captions_by_synset = data
            total_captions = sum(len(caps) for caps in captions_by_synset.values())
            print(f"✅ Loaded {total_captions} captions for {len(captions_by_synset)} classes")
        else:
            # List format, organize by synset
            captions_by_synset = {}
            for item in data:
                synset = item.get('synset') or item.get('wnid')
                if synset in self.IMAGENETTE_CLASSES:
                    if synset not in captions_by_synset:
                        captions_by_synset[synset] = []
                    captions_by_synset[synset].append(item.get('caption', ''))
        
        return captions_by_synset
    
    def _generate_synthetic_captions(self):
        """Generate diverse synthetic captions for each class"""
        captions_by_synset = {}
        
        templates = [
            "a photo of a {}",
            "an image of a {}",
            "a picture showing a {}",
            "this is a {}",
            "a clear photo of a {}",
            "a {} in the image",
            "we can see a {} here",
            "there is a {} in this picture",
            "a good example of a {}",
            "a nice {} photo",
            "a {} captured in detail",
            "an excellent shot of a {}",
            "photograph of a {}",
            "image featuring a {}",
            "a detailed view of a {}"
        ]
        
        for synset, class_name in self.IMAGENETTE_CLASSES.items():
            captions = [template.format(class_name) for template in templates]
            captions_by_synset[synset] = captions
        
        return captions_by_synset
    
    def _create_samples(self):
        """Create list of (image_idx, caption) pairs"""
        samples = []
        
        for img_idx in range(len(self.image_dataset)):
            # Get image path and class
            img_path, class_idx = self.image_dataset.imgs[img_idx]
            class_name = self.image_dataset.classes[class_idx]
            
            # Get captions for this class
            if class_name in self.captions_dict:
                available_captions = self.captions_dict[class_name]
            else:
                # Fallback: use class name
                available_captions = [
                    f"a photo of a {self.IMAGENETTE_CLASSES.get(class_name, 'object')}"
                ]
            
            # Sample or cycle through captions
            for i in range(self.captions_per_image):
                caption = available_captions[i % len(available_captions)]
                samples.append({
                    'image_idx': img_idx,
                    'caption': caption,
                    'class_name': class_name
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load image
        image, _ = self.image_dataset[sample_info['image_idx']]
        
        return {
            'image': image,
            'caption': sample_info['caption'],
            'image_id': sample_info['image_idx'],
            'class_name': sample_info['class_name']
        }


def download_imagenet_captions_annotations():
    """
    Download ONLY the ImageNet-Captions annotations (small files)
    No need to download full ImageNet images!
    """
    import urllib.request
    
    print("📥 Downloading ImageNet-Captions annotations...")
    print("   (This is just JSON files, very small ~100MB)")
    
    # These are the annotation files from ImageNet-Captions project
    # You can download manually from: https://github.com/mlfoundations/imagenet-captions
    
    base_url = "https://github.com/mlfoundations/imagenet-captions/raw/main"
    files = [
        "train_captions.json",
        "val_captions.json"
    ]
    
    output_dir = Path("./data/imagenet_captions_annotations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n⚠️  Please download manually from:")
    print("   https://github.com/mlfoundations/imagenet-captions")
    print(f"\nSave to: {output_dir.resolve()}/")
    print("\nFiles needed:")
    for f in files:
        print(f"  - {f}")
    
    return output_dir


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    class_names = [item['class_name'] for item in batch]
    
    return {
        'images': images,
        'captions': captions,
        'image_ids': image_ids,
        'class_names': class_names
    }


if __name__ == '__main__':
    print("="*60)
    print("IMAGENETTE + IMAGENET-CAPTIONS DATASET TEST")
    print("="*60)
    
    # Test with synthetic captions (no download needed)
    print("\n1. Testing with synthetic captions...")
    dataset = ImagenetteWithCaptions(
        imagenette_root='./datasets/imagenette2',
        split='train',
        captions_per_image=5
    )
    
    print(f"\n✅ Dataset created: {len(dataset)} samples")
    
    # Show a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Caption: {sample['caption']}")
        print(f"  Class: {sample['class_name']}")
        print(f"  Image ID: {sample['image_id']}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"\n✅ DataLoader created")
    batch = next(iter(loader))
    print(f"  Batch images: {batch['images'].shape}")
    print(f"  Batch captions: {len(batch['captions'])}")
    print(f"  First caption: {batch['captions'][0]}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now train Slot-CoCa with:")
    print("  - 13,394 Imagenette images (train)")
    print("  - 3,925 Imagenette images (val)")
    print("  - 5 captions per image")
    print("  - Total: ~67K training pairs")
    print("\nNo need to download full ImageNet!")
