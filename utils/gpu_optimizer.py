"""
GPU Optimization utilities for TensorFlow

FILE PURPOSE:
    Provides comprehensive GPU optimization utilities for TensorFlow deep learning.
    Handles GPU memory management, mixed precision training, TensorFlow optimizations,
    optimal batch size detection, and data pipeline optimization for maximum performance.

DEPENDENCIES (IMPORTS FROM):
    - tensorflow: Core deep learning framework for GPU operations
    - os: Environment variable management
    - warnings: Warning suppression for cleaner output

OUTPUTS (GENERATES):
    - GPU configuration: Memory growth, mixed precision, XLA compilation settings
    - Optimal batch size: Automatically detected best batch size for GPU
    - Optimized data pipeline: Cached, prefetched, and batched data pipeline
    - GPU status information: Memory usage, device capabilities, performance metrics

ROLE IN PROJECT:
    Performance optimization component that maximizes GPU utilization.
    Used by main.py to set up GPU environment and optimize training performance.
    Ensures efficient use of NVIDIA GPU resources for faster training and inference.
"""

import tensorflow as tf
import os
import warnings
import logging

# Suppress TensorFlow mixed precision warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO, WARNING, and ERROR messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

# Suppress specific mixed precision warnings
tf.get_logger().setLevel("CRITICAL")


def configure_gpu_memory_growth():
    """
    Configure GPU memory growth to avoid allocating all GPU memory at once
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")
            return True
        else:
            print("⚠️  No GPUs detected, using CPU")
            return False
    except Exception as e:
        print(f"❌ Error configuring GPU memory: {e}")
        return False


def set_mixed_precision_policy():
    """
    Enable mixed precision training for better performance on modern GPUs
    """
    try:
        # Check if GPU supports mixed precision
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Enable mixed precision
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision (float16) enabled")
            return True
        else:
            print("⚠️  No GPU detected, using float32 precision")
            return False
    except Exception as e:
        print(f"❌ Error setting mixed precision: {e}")
        return False


def configure_tensorflow_optimizations():
    """
    Configure TensorFlow for optimal performance
    """
    # Enable XLA compilation for faster execution
    tf.config.optimizer.set_jit(True)

    # Enable experimental optimizations
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
            "debug_stripper": True,
            "scoped_allocator_optimization": True,
            "pin_to_host_optimization": True,
            "implementation_selector": True,
            "auto_mixed_precision": True,
            "disable_meta_optimizer": False,
            "min_graph_nodes": 2,
        }
    )

    print("✅ TensorFlow optimizations enabled")


def get_optimal_batch_size(model, sample_data, max_batch_size=512):
    """
    Find optimal batch size for the given model and data
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return 32  # Default for CPU

        # Start with a reasonable batch size
        batch_size = 64
        max_found = False

        while batch_size <= max_batch_size and not max_found:
            try:
                # Test if this batch size works
                test_batch = sample_data[:batch_size]
                _ = model(test_batch, training=False)
                batch_size *= 2
            except tf.errors.ResourceExhaustedError:
                max_found = True
                batch_size = batch_size // 2
            except Exception:
                max_found = True
                batch_size = batch_size // 2

        # Ensure minimum batch size
        batch_size = max(batch_size, 16)
        print(f"✅ Optimal batch size determined: {batch_size}")
        return batch_size

    except Exception as e:
        print(f"⚠️  Could not determine optimal batch size: {e}")
        return 32


def create_optimized_data_pipeline(dataset, batch_size, prefetch=True, cache=True):
    """
    Create an optimized data pipeline
    """
    # try:
    #     # Cache dataset if it fits in memory
    #     if cache:
    #         dataset = dataset.cache()

    #     # Shuffle and batch
    #     dataset = dataset.shuffle(buffer_size=1000)
    #     dataset = dataset.batch(batch_size)

    #     # Prefetch for better performance
    #     if prefetch:
    #         dataset = dataset.prefetch(tf.data.AUTOTUNE)

    #     print("✅ Optimized data pipeline created")
    #     return dataset

    # except Exception as e:
    #     print(f"❌ Error creating optimized data pipeline: {e}")
    #     return dataset

    try:
        # Cache dataset if it fits in memory
        if cache:
            dataset = dataset.cache()

        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)

        # Prefetch for better performance
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        print("✅ Optimized data pipeline created")
        return dataset

    except Exception as e:
        print(f"❌ Error creating optimized data pipeline: {e}")
        return dataset


def setup_gpu_environment():
    """
    Complete GPU environment setup
    """
    print("Setting up GPU environment...")
    print("=" * 50)

    # Configure GPU memory growth
    gpu_available = configure_gpu_memory_growth()

    # Set mixed precision if GPU available
    if gpu_available:
        set_mixed_precision_policy()

    # Configure TensorFlow optimizations
    configure_tensorflow_optimizations()

    # Set environment variables for better performance
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 50)
    return gpu_available


def print_gpu_info():
    """
    Print detailed GPU information
    """
    print("\nGPU Information:")
    print("=" * 30)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            print(f"  Device Type: {gpu.device_type}")

            # Try to get memory info
            try:
                # This requires nvidia-ml-py
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                print(f"  Total Memory: {info.total / 1024**3:.2f} GB")
                print(f"  Free Memory: {info.free / 1024**3:.2f} GB")
                print(f"  Used Memory: {info.used / 1024**3:.2f} GB")
            except ImportError:
                print("  Memory info not available (install nvidia-ml-py)")
            except Exception as e:
                print(f"  Memory info error: {e}")
    else:
        print("No GPUs detected")

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"CUDA Available: {tf.test.is_built_with_cuda()}")

    # Check if GPU is available for computation
    try:
        gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
        print(f"GPU Computation: {'Available' if gpu_available else 'Not Available'}")
    except:
        print("GPU Computation: Unknown")


def fix_driver_issue():
    """
    Provide instructions to fix NVIDIA driver issues
    """
    print("\n" + "=" * 60)
    print("NVIDIA DRIVER ISSUE DETECTED")
    print("=" * 60)
    print("Your system has a driver version mismatch:")
    print("- NVIDIA driver version: 580.65.6")
    print("- Kernel driver version: 575.64.3")
    print("\nTo fix this issue, run the following commands:")
    print("\n1. Update your system:")
    print("   sudo apt update && sudo apt upgrade")
    print("\n2. Install the latest NVIDIA drivers:")
    print("   sudo apt install nvidia-driver-580")
    print("\n3. Reboot your system:")
    print("   sudo reboot")
    print("\n4. After reboot, verify the fix:")
    print("   nvidia-smi")
    print("\n5. Then run your training again with GPU acceleration!")
    print("=" * 60)
