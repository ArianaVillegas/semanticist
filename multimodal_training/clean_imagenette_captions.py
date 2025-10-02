"""
Clean ImageNet-Captions to remove low-quality entries
Filters out image IDs, short captions, etc.
"""
import json
from pathlib import Path

def is_good_caption(caption, min_words=3):
    """Check if caption is good quality"""
    # Remove whitespace
    caption = caption.strip()
    
    # Filter criteria
    if len(caption) < 5:
        return False
    
    # Check if it's mostly numbers (likely an ID)
    digits = sum(c.isdigit() for c in caption)
    if digits / len(caption) > 0.5:
        return False
    
    # Check word count
    words = caption.split()
    if len(words) < min_words:
        return False
    
    # Filter common bad patterns
    bad_patterns = [
        '_',  # Filenames often have underscores
        '.jpg', '.png', '.jpeg',  # File extensions
        'IMG_', 'DSC_', 'DSCN',  # Camera defaults
    ]
    
    if any(pattern in caption for pattern in bad_patterns):
        return False
    
    return True

def clean_captions(input_file, output_file, min_words=3):
    """Clean caption file"""
    print("="*60)
    print("CLEANING IMAGENETTE CAPTIONS")
    print("="*60)
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📂 Input: {input_file}")
    print(f"   Total captions: {sum(len(caps) for caps in data.values())}")
    
    # Clean each class
    cleaned_data = {}
    stats = {'total_before': 0, 'total_after': 0, 'removed': 0}
    
    for synset, captions in data.items():
        stats['total_before'] += len(captions)
        
        # Filter captions
        good_captions = [cap for cap in captions if is_good_caption(cap, min_words)]
        
        stats['total_after'] += len(good_captions)
        stats['removed'] += len(captions) - len(good_captions)
        
        if good_captions:
            cleaned_data[synset] = good_captions
        
        print(f"\n{synset}: {len(captions)} → {len(good_captions)} captions")
        if len(good_captions) > 0:
            print(f"  ✅ Sample: {good_captions[0][:60]}")
        if len(captions) - len(good_captions) > 0:
            print(f"  ❌ Removed: {len(captions) - len(good_captions)} low-quality")
    
    # Save cleaned data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ CLEANING COMPLETE")
    print("="*60)
    print(f"\n📊 Statistics:")
    print(f"   Before: {stats['total_before']} captions")
    print(f"   After: {stats['total_after']} captions")
    print(f"   Removed: {stats['removed']} low-quality")
    print(f"   Quality: {stats['total_after']/stats['total_before']*100:.1f}%")
    print(f"\n💾 Saved to: {output_file}")

if __name__ == '__main__':
    clean_captions(
        input_file='data/imagenette_captions/imagenette_captions.json',
        output_file='data/imagenette_captions/imagenette_captions_clean.json',
        min_words=3
    )
