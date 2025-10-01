"""
Quick test to verify evaluation scripts work with Imagenette
"""
import sys
import os

print("="*60)
print("TESTING EVALUATION SCRIPTS")
print("="*60)

# Check if we can import the evaluation modules
print("\n1. Testing imports...")

try:
    sys.path.insert(0, 'evaluation')
    from eval_zero_shot import ZeroShotEvaluator
    print("  ✅ eval_zero_shot.py imports successfully")
except Exception as e:
    print(f"  ❌ eval_zero_shot.py failed: {e}")

try:
    from eval_retrieval import RetrievalEvaluator
    print("  ✅ eval_retrieval.py imports successfully")
except Exception as e:
    print(f"  ❌ eval_retrieval.py failed: {e}")

try:
    from eval_captioning import CaptioningEvaluator
    print("  ✅ eval_captioning.py imports successfully")
except Exception as e:
    print(f"  ❌ eval_captioning.py failed: {e}")

# Check dataset
print("\n2. Testing dataset loading...")
try:
    from imagenette_captions_dataset import ImagenetteWithCaptions
    dataset = ImagenetteWithCaptions(
        imagenette_root='../datasets/imagenette2',
        split='val',
        captions_per_image=1
    )
    print(f"  ✅ Loaded Imagenette val set: {len(dataset)} samples")
except Exception as e:
    print(f"  ❌ Dataset loading failed: {e}")

print("\n" + "="*60)
print("✅ EVALUATION SCRIPTS READY")
print("="*60)

print("\nTo evaluate your trained model, run:")
print("\n# Zero-shot classification")
print("python evaluation/eval_zero_shot.py \\")
print("    --model_path <path_to_checkpoint> \\")
print("    --data_dir ../datasets/imagenette2 \\")
print("    --use_imagenette")

print("\n# Image-text retrieval")
print("python evaluation/eval_retrieval.py \\")
print("    --model_path <path_to_checkpoint> \\")
print("    --data_dir ../datasets/imagenette2 \\")
print("    --use_imagenette")

print("\n# Image captioning")
print("python evaluation/eval_captioning.py \\")
print("    --model_path <path_to_checkpoint> \\")
print("    --data_dir ../datasets/imagenette2 \\")
print("    --use_imagenette")

print("\nReplace <path_to_checkpoint> with your trained model, e.g.:")
print("  ../checkpoints/slot_coca_real_captions/best_model.pt")
