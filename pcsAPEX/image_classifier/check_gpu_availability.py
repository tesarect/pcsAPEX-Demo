"""
Utility script to check GPU availability for TensorFlow
"""

import tensorflow as tf
import sys

def check_gpu():
    """Check if GPU is available and properly configured for TensorFlow"""
    print("TensorFlow version:", tf.__version__)
    
    # Check for physical GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Physical GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    if not gpus:
        print("No GPUs detected. TensorFlow will use CPU only.")
        return
    
    # Try to configure memory growth
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu}")
        except Exception as e:
            print(f"Error configuring {gpu}: {e}")
    
    # Test GPU with a simple operation
    print("\nTesting GPU with a simple operation...")
    try:
        with tf.device('/GPU:0'):
            x = tf.random.normal([1000, 1000])
            result = tf.reduce_sum(x)
            print(f"GPU test successful. Result: {result}")
    except Exception as e:
        print(f"GPU test failed: {e}")
        print("TensorFlow will likely fall back to CPU.")

if __name__ == "__main__":
    check_gpu()
    sys.exit(0)