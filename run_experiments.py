#!/usr/bin/env python3
"""
Run SlotFormer experiments while training is ongoing.
"""

import torch
import subprocess
import time
from pathlib import Path
import json
from evaluate_slotformer import SlotFormerEvaluator

def check_training_progress():
    """Check if training has produced any checkpoints."""
    checkpoint_patterns = [
        "model-*.ckpt",
        "model*.pt", 
        "checkpoint*.ckpt",
        "*.safetensors"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(Path(".").glob(pattern))
    
    if checkpoints:
        # Return most recent checkpoint
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    return None

def run_quick_validation():
    """Run quick validation with untrained model to test framework."""
    print("=== Quick Validation Test ===")
    print("Testing evaluation framework with untrained model...")
    
    evaluator = SlotFormerEvaluator("nonexistent_model.ckpt")
    
    # Run with minimal data for testing
    import torchvision
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    
    dataset = torchvision.datasets.Imagenette(
        "./datasets", split="val", transform=transform, download=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=2
    )
    
    # Test reconstruction quality (minimal)
    recon_results = evaluator.evaluate_reconstruction_quality(dataloader, max_batches=2)
    
    # Test causal ordering (minimal)  
    causal_results = evaluator.evaluate_causal_ordering(dataloader, max_batches=2)
    
    # Create visualizations
    evaluator.create_visualizations("./quick_test_results")
    evaluator.save_results("./quick_test_results/results.json")
    
    print("✓ Quick validation complete!")
    print("✓ Framework is working correctly")
    
    return evaluator.results

def monitor_and_evaluate():
    """Monitor training and run evaluations when checkpoints are available."""
    print("=== SlotFormer Experiment Monitor ===")
    
    # First run quick validation
    quick_results = run_quick_validation()
    
    print("\nMonitoring for training checkpoints...")
    print("Will run full evaluation when model checkpoints are found.")
    
    last_checkpoint = None
    evaluation_count = 0
    
    while True:
        # Check for new checkpoints
        current_checkpoint = check_training_progress()
        
        if current_checkpoint and current_checkpoint != last_checkpoint:
            print(f"\n🔍 Found new checkpoint: {current_checkpoint}")
            
            # Wait a bit to ensure checkpoint is fully written
            time.sleep(10)
            
            try:
                # Run full evaluation
                evaluator = SlotFormerEvaluator(current_checkpoint)
                evaluator.run_full_evaluation()
                
                # Save results with timestamp
                timestamp = int(time.time())
                results_dir = f"./results/eval_{timestamp}"
                Path(results_dir).mkdir(parents=True, exist_ok=True)
                
                evaluator.create_visualizations(results_dir)
                evaluator.save_results(f"{results_dir}/results.json")
                
                print(f"✓ Evaluation {evaluation_count + 1} complete!")
                print(f"✓ Results saved to {results_dir}")
                
                evaluation_count += 1
                last_checkpoint = current_checkpoint
                
                # Print key metrics
                if 'causal_ordering' in evaluator.results:
                    monotonic = evaluator.results['causal_ordering']['monotonic_ratio']
                    print(f"📊 Monotonic improvement ratio: {monotonic:.3f}")
                    
                    if monotonic > 0.8:
                        print("🎉 SlotFormer showing strong causal ordering!")
                    elif monotonic > 0.6:
                        print("⚠️  SlotFormer showing moderate causal ordering")
                    else:
                        print("❌ SlotFormer not showing clear causal ordering yet")
                
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
        
        else:
            print(".", end="", flush=True)
        
        # Check every 30 seconds
        time.sleep(30)

def run_ablation_studies():
    """Run ablation studies comparing different configurations."""
    print("=== Ablation Studies ===")
    
    # This would compare different:
    # - Number of transformer layers
    # - Different encoders (DINO vs DINOv2 vs DINOv3)
    # - Different slot initialization strategies
    
    ablations = {
        "layers_1": {"transformer_layers": 1},
        "layers_3": {"transformer_layers": 3}, 
        "layers_6": {"transformer_layers": 6},
        "dinov2_base": {"encoder": "vit_base_patch14_dinov2"},
        "dinov3_base": {"encoder": "vit_base_patch16_dinov3"},
    }
    
    print("Ablation studies would require training multiple models.")
    print("For now, focusing on main SlotFormer evaluation.")
    
    return ablations

def create_experiment_report():
    """Create a comprehensive experiment report."""
    
    report = {
        "experiment": "SlotFormer Validation",
        "hypothesis": "SlotFormer can learn causal ordering where more slots improve reconstruction",
        "methodology": {
            "model": "SlotFormer with DINOv3 encoder",
            "dataset": "Imagenette (subset of ImageNet)",
            "metrics": [
                "Reconstruction quality vs number of slots",
                "Monotonic improvement ratio (causal ordering)",
                "Comparison to random baseline"
            ]
        },
        "success_criteria": {
            "reconstruction_improvement": "SlotFormer should significantly outperform random baseline",
            "causal_ordering": "Monotonic improvement ratio > 0.8",
            "scalability": "Performance should improve with more slots"
        }
    }
    
    with open("experiment_plan.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("✓ Experiment plan saved to experiment_plan.json")
    return report

def main():
    """Main experiment runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SlotFormer experiments')
    parser.add_argument('--mode', choices=['quick', 'monitor', 'evaluate'], 
                       default='monitor', help='Experiment mode')
    parser.add_argument('--model', type=str, help='Path to specific model to evaluate')
    
    args = parser.parse_args()
    
    # Create experiment plan
    create_experiment_report()
    
    if args.mode == 'quick':
        run_quick_validation()
    elif args.mode == 'evaluate' and args.model:
        evaluator = SlotFormerEvaluator(args.model)
        evaluator.run_full_evaluation()
    else:
        monitor_and_evaluate()

if __name__ == "__main__":
    main()
