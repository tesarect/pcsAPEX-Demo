#!/usr/bin/env python
"""
Check GPU availability and compatibility with TensorFlow
"""
import sys
from image_classifier.tf_config import check_tf_cuda_compatibility, test_gpu_performance

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check GPU compatibility with TensorFlow")
    parser.add_argument("--test", action="store_true", help="Run performance test")
    args = parser.parse_args()
    
    # Always check compatibility
    check_tf_cuda_compatibility()
    
    # Run performance test if requested
    if args.test:
        print("\n" + "="*50)
        print("Running GPU performance test...")
        print("="*50 + "\n")
        test_gpu_performance()
    
    sys.exit(0)