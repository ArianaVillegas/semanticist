#!/usr/bin/env python3
"""
Debug conda environment issues.
"""

import sys
import os

def debug_environment():
    """Debug Python and conda environment setup."""
    
    print("=== Python Environment Debug ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    
    print(f"\n=== Environment Variables ===")
    print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")
    print(f"PATH: {os.environ.get('PATH', 'Not set')}")
    
    print(f"\n=== Expected vs Actual ===")
    expected_python = "/home/avillegas/miniconda3/envs/semanticist/bin/python"
    actual_python = sys.executable
    
    print(f"Expected Python: {expected_python}")
    print(f"Actual Python: {actual_python}")
    print(f"Match: {expected_python == actual_python}")
    
    # Check if timm is in the right environment
    print(f"\n=== Package Location Check ===")
    try:
        import timm
        print(f"✓ timm found at: {timm.__file__}")
        print(f"timm version: {timm.__version__}")
    except ImportError as e:
        print(f"✗ timm import failed: {e}")
        
        # Check if it's in the expected location
        expected_timm = "/home/avillegas/miniconda3/envs/semanticist/lib/python3.13/site-packages/timm"
        if os.path.exists(expected_timm):
            print(f"✓ timm exists at expected location: {expected_timm}")
        else:
            print(f"✗ timm not found at expected location: {expected_timm}")

if __name__ == "__main__":
    debug_environment()
