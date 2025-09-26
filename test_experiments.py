#!/usr/bin/env python3
"""
Test script to verify all experiments can import correctly.
"""

import sys
import os

def test_imports():
    """Test that all experiment modules can be imported."""
    
    print("=== Testing Experiment Imports ===")
    
    try:
        # Test scalability study
        sys.path.append('./experiments')
        from scalability_study import ScalabilityStudy
        print("✓ ScalabilityStudy import successful")
        
        # Test causal validation
        from causal_validation import CausalValidationExperiments
        print("✓ CausalValidationExperiments import successful")
        
        # Test multimodal extension
        from multimodal_extension import MultiModalSlotFormer, MultiModalExperiments
        print("✓ MultiModalSlotFormer import successful")
        
        print("\n✅ All experiment imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of experiment classes."""
    
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Test that we can create instances
        from experiments.scalability_study import ScalabilityStudy
        study = ScalabilityStudy(device="cpu")  # Use CPU for testing
        print("✓ ScalabilityStudy instance created")
        
        from experiments.causal_validation import CausalValidationExperiments
        validator = CausalValidationExperiments(device="cpu")
        print("✓ CausalValidationExperiments instance created")
        
        from experiments.multimodal_extension import MultiModalExperiments
        multimodal = MultiModalExperiments(device="cpu")
        print("✓ MultiModalExperiments instance created")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("Testing SlotFormer Experiments")
    print("=" * 40)
    
    # Test imports
    import_success = test_imports()
    
    if import_success:
        # Test basic functionality
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n🎉 All tests passed! Experiments are ready to run.")
            return 0
        else:
            print("\n⚠️ Import tests passed but functionality tests failed.")
            return 1
    else:
        print("\n❌ Import tests failed. Check your Python path and dependencies.")
        return 1

if __name__ == "__main__":
    exit(main())
