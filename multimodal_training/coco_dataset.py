"""
COCO Captions Dataset for Slot-CoCa
Much more reliable than ImageNet-Captions
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
from pathlib import Path
from pycocotools.coco import COCO

class COCOCaptionsDataset(Dataset):
    """
    COCO Captions dataset
    
    - 123K training images
    - 5 captions per image (human-written, high quality)
    - Perfect for multimodal learning
    """
    
    def __init__(
        self,
        root_dir,
        ann_file,
        transform=None,
        max_caption_length=77,
        captions_per_image=5
    ):
        """
        Args:
            root_dir: Path to COCO images directory
            ann_file: Path to COCO annotations JSON
            transform: Image transforms
            max_caption_length: Max tokens for captions
            captions_per_image: Number of captions per image
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()
        self.max_caption_length = max_caption_length
        self.captions_per_image = captions_per_image
        
        # Load COCO API
        print(f"Loading COCO annotations from {ann_file}...")
        self.coco = COCO(ann_file)
        
        # Get all image IDs
        self.img_ids = list(self.coco.imgs.keys())
        
        # Create image-caption pairs
        self.samples = self._create_samples()
        
        print(f"✅ Loaded {len(self.samples)} image-caption pairs")
        print(f"   - Unique images: {len(self.img_ids)}")
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
    
    def _create_samples(self):
        """Create list of (image_id, caption) pairs"""
        samples = []
        
        for img_id in self.img_ids:
            # Get captions for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Take up to captions_per_image captions
            captions = [ann['caption'] for ann in anns[:self.captions_per_image]]
            
            # Pad with duplicates if needed
            while len(captions) < self.captions_per_image:
                captions.append(captions[0] if captions else "")
            
            for caption in captions[:self.captions_per_image]:
                samples.append({
                    'image_id': img_id,
                    'caption': caption
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        img_id = sample_info['image_id']
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root_dir / img_info['file_name']
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'caption': sample_info['caption'],
            'image_id': img_id
        }


def collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'captions': captions,
        'image_ids': image_ids
    }


if __name__ == '__main__':
    print("="*60)
    print("COCO CAPTIONS DATASET TEST")
    print("="*60)
    
    # Test dataset
    dataset = COCOCaptionsDataset(
        root_dir='../datasets/coco/train2017',
        ann_file='../datasets/coco/annotations/captions_train2017.json',
        captions_per_image=5
    )
    
    print(f"\nDataset created: {len(dataset)} samples")
    
    # Show samples
    print("\n📝 Sample captions:")
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        print(f"\nImage {sample['image_id']}:")
        print(f"  Caption: {sample['caption']}")
        print(f"  Image shape: {sample['image'].shape}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(loader))
    print(f"\n✅ Batch loaded:")
    print(f"  Images: {batch['images'].shape}")
    print(f"  Captions: {len(batch['captions'])}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
