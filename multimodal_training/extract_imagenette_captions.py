"""
Extract ImageNet-Captions for Imagenette classes only

This script processes the full ImageNet-Captions dataset and extracts
only the captions for the 10 Imagenette classes, making it much smaller
and faster to use.
"""
import json
from pathlib import Path
from collections import defaultdict
import argparse

# Imagenette classes (10 classes subset of ImageNet)
IMAGENETTE_SYNSETS = {
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

def extract_imagenette_captions(imagenet_captions_dir, output_dir):
    """
    Extract captions for Imagenette classes from ImageNet-Captions
    
    Args:
        imagenet_captions_dir: Path to cloned imagenet-captions repo
        output_dir: Where to save extracted captions
    """
    imagenet_captions_dir = Path(imagenet_captions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXTRACTING IMAGENETTE CAPTIONS")
    print("="*60)
    print(f"Source: {imagenet_captions_dir}")
    print(f"Output: {output_dir}")
    print(f"Target classes: {len(IMAGENETTE_SYNSETS)}")
    print("="*60)
    
    # Check if source exists
    if not imagenet_captions_dir.exists():
        print(f"\n❌ Error: {imagenet_captions_dir} not found!")
        print("Please run:")
        print("  git clone https://github.com/mlfoundations/imagenet-captions.git /tmp/imagenet-captions")
        return
    
    # Look for the main captions file
    captions_file = imagenet_captions_dir / 'imagenet_captions.json'
    
    if not captions_file.exists():
        print(f"\n❌ Error: {captions_file} not found!")
        print("\nFiles in directory:")
        for item in imagenet_captions_dir.iterdir():
            print(f"  - {item.name}")
        return
    
    print(f"\n📂 Loading captions from {captions_file}...")
    print("   (This may take a moment, file is ~131MB)")
    
    # Load all captions
    try:
        with open(captions_file, 'r') as f:
            all_data = json.load(f)
        
        print(f"   ✅ Loaded {len(all_data)} total image entries")
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return
    
    # The format is typically: list of {image_id, synset, captions: [...]}
    # Extract Imagenette captions
    imagenette_captions = defaultdict(list)
    
    print("\n🔍 Extracting Imagenette captions...")
    
    for item in all_data:
        # Get synset/class ID
        synset = item.get('wnid') or item.get('synset') or item.get('class_id')
        
        # Add to our collection if it's an Imagenette class
        if synset in IMAGENETTE_SYNSETS:
            # Collect captions from different possible fields
            potential_captions = []
            
            # Try different fields
            if item.get('title'):
                potential_captions.append(item['title'])
            if item.get('description'):
                potential_captions.append(item['description'])
            if item.get('caption'):
                potential_captions.append(item['caption'])
            if item.get('captions'):
                caps = item['captions']
                if isinstance(caps, list):
                    potential_captions.extend(caps)
                else:
                    potential_captions.append(caps)
            
            # Add all non-empty captions
            for caption in potential_captions:
                if caption and isinstance(caption, str) and len(caption.strip()) > 0:
                    imagenette_captions[synset].append(caption.strip())
    
    if not imagenette_captions:
        print("\n⚠️  No Imagenette captions found!")
        print("Checking data format...")
        if all_data:
            print(f"Sample entry: {json.dumps(all_data[0], indent=2)[:500]}")
        return
    
    # Save extracted captions (single file with all captions)
    output_file = output_dir / 'imagenette_captions.json'
    
    with open(output_file, 'w') as f:
        json.dump(dict(imagenette_captions), f, indent=2)
    
    # Statistics
    total_captions = sum(len(caps) for caps in imagenette_captions.values())
    print(f"\n✅ Extracted {total_captions} captions for {len(imagenette_captions)} Imagenette classes")
    print(f"   Saved to: {output_file}")
    
    # Show per-class stats
    print(f"\n📊 Per-class breakdown:")
    for synset, class_name in sorted(IMAGENETTE_SYNSETS.items()):
        count = len(imagenette_captions.get(synset, []))
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {class_name:20s} ({synset}): {count:5d} captions")
    
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE")
    print("="*60)
    print(f"\nCaptions saved to: {output_dir}/")
    print("\nUsage:")
    print(f"  python imagenette_captions_dataset.py \\")
    print(f"      --captions_file {output_dir}/imagenette_train_captions.json")


def main():
    parser = argparse.ArgumentParser(description='Extract Imagenette captions from ImageNet-Captions')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/tmp/imagenet-captions',
        help='Path to cloned imagenet-captions repository'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/imagenette_captions',
        help='Where to save extracted captions'
    )
    
    args = parser.parse_args()
    
    extract_imagenette_captions(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
