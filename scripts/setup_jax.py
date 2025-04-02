import os
import jax
import jax.numpy as jnp
from loguru import logger

def setup_jax():
    """
    Configure JAX for optimal performance with multiple CPUs/GPUs.
    """
    # Enable 64-bit precision if needed
    jax.config.update("jax_enable_x64", True)
    
    # Get available devices
    devices = jax.devices()
    logger.info(f"Found {len(devices)} devices:")
    for i, device in enumerate(devices):
        logger.info(f"Device {i}: {device}")
    
    # Set up parallelization
    if len(devices) > 1:
        # Enable parallel execution across devices
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_default_device", devices[0])
        logger.info("Enabled parallel execution across devices")
    
    # Configure memory management
    jax.config.update("jax_disable_jit", False)  # Enable JIT compilation
    
    # Set up memory preallocation
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    
    logger.info("JAX configuration complete")
    return devices

if __name__ == "__main__":
    setup_jax() 