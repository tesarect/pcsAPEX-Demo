"""
TensorFlow GPU configuration utilities
"""
import os
import logging
import tensorflow as tf
import subprocess
import sys

logger = logging.getLogger(__name__)

def configure_tensorflow(environment="auto"):
    """
    Configure TensorFlow based on environment and GPU availability
    
    Args:
        environment: One of "auto", "dev", "test", "prod", or "cpu_only"
    
    Returns:
        dict: Configuration information
    """
    # Configure TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    try:
        # Check for available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        # Force CPU-only mode if requested
        if environment == "cpu_only":
            tf.config.set_visible_devices([], 'GPU')
            logger.info("Forced CPU-only mode as requested")
            return {"mode": "cpu", "gpus": 0, "reason": "forced"}
        
        # No GPUs available
        if not gpus:
            logger.info("No GPUs detected, using CPU")
            return {"mode": "cpu", "gpus": 0, "reason": "not_available"}
        
        # Configure memory growth to prevent OOM errors
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Enabled memory growth for GPU: {gpu}")
            except RuntimeError as e:
                logger.warning(f"Error configuring GPU memory growth: {e}")
        
        # Enable soft device placement to fall back to CPU for ops without GPU implementation
        tf.config.set_soft_device_placement(True)
        
        # Enable deterministic operations for testing
        if environment == "test":
            try:
                tf.config.experimental.enable_op_determinism()
                logger.info("Enabled deterministic operations for testing")
            except Exception as e:
                logger.warning(f"Could not enable deterministic operations: {e}")
        
        # Initialize TensorFlow by creating and executing a small operation
        # This forces TensorFlow to initialize its internal state
        with tf.device('/GPU:0'):
            tf.random.normal([100, 100])
        
        logger.info(f"TensorFlow initialized with {len(gpus)} GPU(s)")
        return {"mode": "gpu", "gpus": len(gpus), "reason": "success"}
    
    except Exception as e:
        logger.error(f"Error initializing TensorFlow: {e}")
        # Fall back to CPU-only mode if initialization fails
        try:
            tf.config.set_visible_devices([], 'GPU')
            logger.info("Fallback: Disabled all GPU devices, using CPU only")
            return {"mode": "cpu", "gpus": 0, "reason": "error", "error": str(e)}
        except:
            logger.error("Critical error initializing TensorFlow")
            return {"mode": "error", "gpus": 0, "reason": "critical_error"}

def check_tf_cuda_compatibility():
    """Check TensorFlow version and CUDA compatibility"""
    print(f"TensorFlow version: {tf.__version__}")
    
    # TensorFlow CUDA compatibility matrix
    compatibility_matrix = {
        "2.15": {"cuda": "12.2", "cudnn": "8.9"},
        "2.14": {"cuda": "11.8", "cudnn": "8.7"},
        "2.13": {"cuda": "11.8", "cudnn": "8.7"},
        "2.12": {"cuda": "11.8", "cudnn": "8.6"},
        "2.11": {"cuda": "11.2", "cudnn": "8.1"},
        "2.10": {"cuda": "11.2", "cudnn": "8.1"},
    }
    
    # Get major.minor version
    tf_version = ".".join(tf.__version__.split(".")[:2])
    
    if tf_version in compatibility_matrix:
        compat = compatibility_matrix[tf_version]
        print(f"Compatible CUDA version: {compat['cuda']}")
        print(f"Compatible cuDNN version: {compat['cudnn']}")
    else:
        print(f"Unknown TensorFlow version compatibility for {tf_version}")
        print("Please check the TensorFlow website for compatibility information")
    
    # Check if TensorFlow can see CUDA
    print("\nChecking if TensorFlow can access GPU:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow can see {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("TensorFlow cannot see any GPUs")
    
    # Try to get CUDA version from system
    print("\nChecking system CUDA installation:")
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"], 
                                             stderr=subprocess.STDOUT).decode()
        print("NVCC found:")
        print(nvcc_output.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        print("NVCC (CUDA compiler) not found in PATH")
    
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], 
                                            stderr=subprocess.STDOUT).decode()
        print("\nNVIDIA driver found:")
        # Extract just the version line
        for line in nvidia_smi.split('\n'):
            if "CUDA Version" in line:
                print(line.strip())
                break
    except (subprocess.SubprocessError, FileNotFoundError):
        print("nvidia-smi not found. NVIDIA driver may not be installed")
    
    # Check if CUDA_HOME or similar environment variables are set
    print("\nCUDA environment variables:")
    for var in ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"]:
        if var in os.environ:
            print(f"{var}={os.environ[var]}")
        else:
            print(f"{var} not set")

def test_gpu_performance():
    """Test if TensorFlow can use GPU and compare performance with CPU"""
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check available devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    print(f"Available CPUs: {len(cpus)}")
    for i, cpu in enumerate(cpus):
        print(f"  CPU {i}: {cpu}")
    
    if not gpus:
        print("No GPUs detected. Test will run on CPU only.")
        return
    
    # Configure memory growth to prevent OOM errors
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error configuring GPU memory growth: {e}")
    
    # Create a large matrix multiplication task
    matrix_size = 5000
    
    # Test on CPU
    print(f"\nRunning {matrix_size}x{matrix_size} matrix multiplication on CPU...")
    with tf.device('/CPU:0'):
        import time
        start_time = time.time()
        
        # Create random matrices
        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])
        
        # Perform matrix multiplication
        c = tf.matmul(a, b)
        
        # Force execution and wait for completion
        result = c.numpy().mean()
        
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.2f} seconds")
    
    # Test on GPU
    print(f"\nRunning {matrix_size}x{matrix_size} matrix multiplication on GPU...")
    try:
        with tf.device('/GPU:0'):
            import time
            start_time = time.time()
            
            # Create random matrices
            a = tf.random.normal([matrix_size, matrix_size])
            b = tf.random.normal([matrix_size, matrix_size])
            
            # Perform matrix multiplication
            c = tf.matmul(a, b)
            
            # Force execution and wait for completion
            result = c.numpy().mean()
            
            gpu_time = time.time() - start_time
            print(f"GPU time: {gpu_time:.2f} seconds")
            
            if cpu_time > 0 and gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"\nGPU is {speedup:.2f}x faster than CPU")
                
                if speedup < 2:
                    print("WARNING: GPU speedup is lower than expected. There might be configuration issues.")
                else:
                    print("GPU is working correctly with TensorFlow!")
    
    except Exception as e:
        print(f"Error running test on GPU: {e}")
        print("TensorFlow cannot use the GPU with the current configuration.")

# Initialize TensorFlow when this module is imported
if __name__ != "__main__":
    configure_tensorflow()