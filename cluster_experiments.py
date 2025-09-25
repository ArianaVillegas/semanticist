#!/usr/bin/env python3
"""
Cluster-optimized experimental framework for SlotFormer.
"""

import torch
import os
import sys
from pathlib import Path
import json
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from comprehensive_evaluation import ComprehensiveEvaluator
from evaluate_slotformer import SlotFormerEvaluator

class ClusterExperimentRunner:
    """Run experiments optimized for cluster environment."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results_dir = Path("./cluster_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"Running on device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def check_environment(self):
        """Check cluster environment setup."""
        print("=== Environment Check ===")
        
        # Check SLURM variables
        slurm_vars = ['SLURM_JOB_ID', 'SLURM_PROCID', 'SLURMD_NODENAME']
        for var in slurm_vars:
            value = os.environ.get(var, 'Not set')
            print(f"{var}: {value}")
        
        # Check conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not set')
        print(f"Conda environment: {conda_env}")
        
        # Check Python packages
        try:
            import timm, torchvision, lightning
            print("✓ All required packages available")
        except ImportError as e:
            print(f"✗ Missing packages: {e}")
            return False
        
        return True
    
    def find_trained_models(self):
        """Find any trained SlotFormer models."""
        model_patterns = ["model*.ckpt", "*.pt", "checkpoint*.ckpt"]
        models = []
        
        for pattern in model_patterns:
            models.extend(Path(".").glob(pattern))
        
        if models:
            # Sort by modification time (newest first)
            models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(f"Found {len(models)} trained models:")
            for model in models:
                mtime = time.ctime(model.stat().st_mtime)
                size = model.stat().st_size / (1024**2)  # MB
                print(f"  - {model} ({size:.1f} MB, {mtime})")
        else:
            print("No trained models found - will use random weights")
        
        return models
    
    def run_baseline_tests(self):
        """Test all baseline models."""
        print("\n=== Testing Baseline Models ===")
        
        try:
            from baseline_models import test_baselines
            test_baselines()
            return True
        except Exception as e:
            print(f"Baseline test failed: {e}")
            return False
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation."""
        print("\n=== Comprehensive Evaluation ===")
        
        try:
            evaluator = ComprehensiveEvaluator(device=self.device)
            evaluator.run_comprehensive_evaluation()
            
            # Move results to cluster results directory
            if Path("comprehensive_results").exists():
                import shutil
                shutil.move("comprehensive_results", self.results_dir / "comprehensive")
            
            return True
        except Exception as e:
            print(f"Comprehensive evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_trained_models(self, models):
        """Evaluate specific trained models."""
        print(f"\n=== Evaluating {len(models)} Trained Models ===")
        
        for i, model_path in enumerate(models):
            print(f"\nEvaluating model {i+1}/{len(models)}: {model_path}")
            
            try:
                evaluator = SlotFormerEvaluator(str(model_path), device=self.device)
                evaluator.run_full_evaluation()
                
                # Move results to timestamped directory
                model_name = model_path.stem
                timestamp = int(time.time())
                model_results_dir = self.results_dir / f"{model_name}_{timestamp}"
                
                if Path("results").exists():
                    import shutil
                    shutil.move("results", model_results_dir)
                
                print(f"✓ Results saved to {model_results_dir}")
                
            except Exception as e:
                print(f"✗ Evaluation of {model_path} failed: {e}")
    
    def create_summary_report(self):
        """Create summary report of all evaluations."""
        print("\n=== Creating Summary Report ===")
        
        summary = {
            "timestamp": time.time(),
            "date": time.ctime(),
            "device": self.device,
            "environment": {
                "slurm_job_id": os.environ.get('SLURM_JOB_ID'),
                "node": os.environ.get('SLURMD_NODENAME'),
                "conda_env": os.environ.get('CONDA_DEFAULT_ENV'),
            },
            "evaluations_completed": []
        }
        
        # Scan results directories
        for result_dir in self.results_dir.iterdir():
            if result_dir.is_dir():
                result_files = list(result_dir.glob("*.json"))
                if result_files:
                    summary["evaluations_completed"].append({
                        "name": result_dir.name,
                        "files": [str(f.name) for f in result_files],
                        "plots": [str(f.name) for f in result_dir.glob("*.png")]
                    })
        
        # Save summary
        summary_path = self.results_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to {summary_path}")
        
        # Print key findings
        self.print_key_findings()
    
    def print_key_findings(self):
        """Print key experimental findings."""
        print("\n" + "="*60)
        print("KEY EXPERIMENTAL FINDINGS")
        print("="*60)
        
        # Look for comprehensive results
        comp_results_path = self.results_dir / "comprehensive" / "results.json"
        if comp_results_path.exists():
            try:
                with open(comp_results_path) as f:
                    results = json.load(f)
                
                if 'causal_ordering' in results:
                    print("\n🎯 CAUSAL ORDERING RESULTS:")
                    causal_results = results['causal_ordering']
                    
                    # Find SlotFormer result
                    if 'slotformer' in causal_results:
                        sf_score = causal_results['slotformer']['mean']
                        print(f"SlotFormer: {sf_score:.3f}")
                        
                        if sf_score >= 0.8:
                            print("🟢 EXCELLENT - Strong causal ordering detected!")
                        elif sf_score >= 0.6:
                            print("🟡 MODERATE - Some causal ordering present")
                        else:
                            print("🔴 POOR - Little to no causal ordering")
                    
                    # Compare to baselines
                    print("\nBaseline Comparisons:")
                    for model_name, result in causal_results.items():
                        if model_name != 'slotformer':
                            score = result.get('mean', 0)
                            print(f"  {model_name}: {score:.3f}")
                
                print(f"\n📊 Full results available in: {self.results_dir}")
                
            except Exception as e:
                print(f"Could not parse results: {e}")
        else:
            print("No comprehensive results found")
        
        print("="*60)
    
    def run_full_cluster_evaluation(self):
        """Run complete evaluation suite on cluster."""
        print("=== SlotFormer Cluster Evaluation Suite ===")
        
        # Environment check
        if not self.check_environment():
            print("Environment check failed!")
            return False
        
        # Find models
        trained_models = self.find_trained_models()
        
        # Run evaluations
        success = True
        
        # 1. Test baselines
        if not self.run_baseline_tests():
            success = False
        
        # 2. Comprehensive evaluation
        if not self.run_comprehensive_evaluation():
            success = False
        
        # 3. Evaluate trained models
        if trained_models:
            self.evaluate_trained_models(trained_models)
        
        # 4. Create summary
        self.create_summary_report()
        
        print(f"\n{'✓ Evaluation completed successfully!' if success else '⚠️ Evaluation completed with some errors'}")
        print(f"Results directory: {self.results_dir.absolute()}")
        
        return success

def main():
    """Main cluster evaluation function."""
    runner = ClusterExperimentRunner()
    runner.run_full_cluster_evaluation()

if __name__ == "__main__":
    main()
