#!/usr/bin/env python3
"""
Test script to verify JAX and PyTorch installations, GPU availability,
and compare performance between JAX and PyTorch.
"""

import time
import numpy as np
import sys
from typing import Tuple, Dict, Any

def check_installations() -> Dict[str, Any]:
    """Check if JAX and PyTorch are properly installed and report device support."""
    print("=" * 60)
    print("CHECKING INSTALLATIONS")
    print("=" * 60)

    results: Dict[str, Any] = {}

    # ── JAX ──────────────────────────────────────────────────────────────
    try:
        import jax
        from jax import devices

        jax_devices = devices()
        device_strs = [str(d) for d in jax_devices]
        # Use the .platform attribute to detect GPU support
        gpu_available = any(d.platform == "gpu" for d in jax_devices)

        results["jax"] = {
            "installed": True,
            "version": jax.__version__,
            "devices": device_strs,
            "gpu_available": gpu_available,
        }

        print(f"✓ JAX installed: {jax.__version__}")
        print(f"  Available devices: {device_strs}")
        print(f"  GPU available: {gpu_available}")

    except ImportError as e:
        results["jax"] = {"installed": False, "error": str(e)}
        print(f"✗ JAX not installed: {e}")

    # ── PyTorch ─────────────────────────────────────────────────────────
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        mps_avail = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        results["torch"] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": cuda_avail,
            "cuda_device_count": torch.cuda.device_count() if cuda_avail else 0,
            "mps_available": mps_avail,
        }

        print(f"✓ PyTorch installed: {torch.__version__}")
        print(f"  CUDA available: {cuda_avail}")
        if cuda_avail:
            print(f"  CUDA devices: {results['torch']['cuda_device_count']}")
        print(f"  MPS available: {mps_avail}")

    except ImportError as e:
        results["torch"] = {"installed": False, "error": str(e)}
        print(f"✗ PyTorch not installed: {e}")

    return results


def test_jax_operations() -> Dict[str, float]:
    """Test basic JAX operations and measure performance."""
    print("\n" + "=" * 60)
    print("TESTING JAX OPERATIONS")
    print("=" * 60)
    
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit, random
        
        # Create test data
        key = random.PRNGKey(42)
        size = 5000
        a = random.normal(key, (size, size))
        b = random.normal(random.split(key)[1], (size, size))
        
        print(f"Testing with {size}x{size} matrices")
        
        # Test matrix multiplication
        @jit
        def matmul_jax(x, y):
            return jnp.dot(x, y)
        
        # Warm up JIT compilation
        _ = matmul_jax(a[:10, :10], b[:10, :10])
        
        # Time matrix multiplication
        start_time = time.time()
        result = matmul_jax(a, b)
        result.block_until_ready()  # Ensure computation is complete
        matmul_time = time.time() - start_time
        
        # Test element-wise operations
        @jit
        def elementwise_ops(x, y):
            return jnp.sin(x) + jnp.cos(y) + jnp.exp(x * 0.1)
        
        # Warm up
        _ = elementwise_ops(a[:10, :10], b[:10, :10])
        
        start_time = time.time()
        result2 = elementwise_ops(a, b)
        result2.block_until_ready()
        elementwise_time = time.time() - start_time
        
        # Test reduction operations
        @jit
        def reduction_ops(x):
            return jnp.sum(x), jnp.mean(x), jnp.std(x)
        
        # Warm up
        _ = reduction_ops(a[:10, :10])
        
        start_time = time.time()
        sums, means, stds = reduction_ops(a)
        # Convert to Python objects to ensure computation is complete
        _ = float(sums), float(means), float(stds)
        reduction_time = time.time() - start_time
        
        times = {
            'matmul': matmul_time,
            'elementwise': elementwise_time,
            'reduction': reduction_time
        }
        
        print(f"✓ Matrix multiplication: {matmul_time:.4f} seconds")
        print(f"✓ Element-wise operations: {elementwise_time:.4f} seconds")
        print(f"✓ Reduction operations: {reduction_time:.4f} seconds")
        
        return times
        
    except Exception as e:
        print(f"✗ JAX operations failed: {e}")
        return {}

def test_torch_operations() -> Dict[str, float]:
    """Test basic PyTorch operations and measure performance."""
    print("\n" + "=" * 60)
    print("TESTING PYTORCH OPERATIONS")
    print("=" * 60)
    
    try:
        import torch
        
        # Set device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device")
        else:
            device = torch.device('cpu')
            print("Using CPU device")
        
        # Create test data
        size = 5000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        print(f"Testing with {size}x{size} matrices on {device}")
        
        # Warm up GPU if available
        if device.type in ['cuda', 'mps']:
            for _ in range(3):
                _ = torch.mm(a[:100, :100], b[:100, :100])
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Test matrix multiplication
        start_time = time.time()
        result = torch.mm(a, b)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        matmul_time = time.time() - start_time
        
        # Test element-wise operations
        start_time = time.time()
        result2 = torch.sin(a) + torch.cos(b) + torch.exp(a * 0.1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elementwise_time = time.time() - start_time
        
        # Test reduction operations
        start_time = time.time()
        sums = torch.sum(a)
        means = torch.mean(a)
        stds = torch.std(a)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        reduction_time = time.time() - start_time
        
        times = {
            'matmul': matmul_time,
            'elementwise': elementwise_time,
            'reduction': reduction_time
        }
        
        print(f"✓ Matrix multiplication: {matmul_time:.4f} seconds")
        print(f"✓ Element-wise operations: {elementwise_time:.4f} seconds")
        print(f"✓ Reduction operations: {reduction_time:.4f} seconds")
        
        return times
        
    except Exception as e:
        print(f"✗ PyTorch operations failed: {e}")
        return {}

def compare_performance(jax_times: Dict[str, float], torch_times: Dict[str, float]) -> None:
    """Compare performance between JAX and PyTorch."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if not jax_times or not torch_times:
        print("Cannot compare - one or both frameworks failed to run tests")
        return
    
    print(f"{'Operation':<20} {'JAX (s)':<12} {'PyTorch (s)':<12} {'Speedup':<12}")
    print("-" * 60)
    
    for op in jax_times:
        if op in torch_times:
            jax_time = jax_times[op]
            torch_time = torch_times[op]
            speedup = torch_time / jax_time if jax_time > 0 else float('inf')
            
            print(f"{op:<20} {jax_time:<12.4f} {torch_time:<12.4f} {speedup:<12.2f}x")
    
    # Overall comparison
    jax_total = sum(jax_times.values())
    torch_total = sum(torch_times.values())
    overall_speedup = torch_total / jax_total if jax_total > 0 else float('inf')
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {jax_total:<12.4f} {torch_total:<12.4f} {overall_speedup:<12.2f}x")
    
    if overall_speedup > 1:
        print(f"\n🚀 JAX is {overall_speedup:.2f}x faster overall!")
    elif overall_speedup < 1:
        print(f"\n🐎 PyTorch is {1/overall_speedup:.2f}x faster overall!")
    else:
        print(f"\n⚖️  JAX and PyTorch performed similarly!")

def test_gradient_computation() -> None:
    """Test gradient computation in both frameworks."""
    print("\n" + "=" * 60)
    print("TESTING GRADIENT COMPUTATION")
    print("=" * 60)
    
    # JAX gradient test
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad, jit
        
        @jit
        def loss_fn_jax(x):
            return jnp.sum(x**2) + jnp.sin(jnp.sum(x))
        
        grad_fn = jit(grad(loss_fn_jax))
        
        x_jax = jnp.array([1.0, 2.0, 3.0])
        
        start_time = time.time()
        for _ in range(1000):
            gradients = grad_fn(x_jax)
        gradients.block_until_ready()
        jax_grad_time = time.time() - start_time
        
        print(f"✓ JAX gradient computation (1000 iterations): {jax_grad_time:.4f} seconds")
        
    except Exception as e:
        print(f"✗ JAX gradient computation failed: {e}")
        jax_grad_time = None
    
    # PyTorch gradient test
    try:
        import torch
        
        def loss_fn_torch(x):
            return torch.sum(x**2) + torch.sin(torch.sum(x))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
        
        start_time = time.time()
        for _ in range(1000):
            loss = loss_fn_torch(x_torch)
            if x_torch.grad is not None:
                x_torch.grad.zero_()
            loss.backward()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        torch_grad_time = time.time() - start_time
        
        print(f"✓ PyTorch gradient computation (1000 iterations): {torch_grad_time:.4f} seconds")
        
    except Exception as e:
        print(f"✗ PyTorch gradient computation failed: {e}")
        torch_grad_time = None
    
    # Compare gradient computation
    if jax_grad_time and torch_grad_time:
        speedup = torch_grad_time / jax_grad_time
        print(f"\nGradient computation speedup: {speedup:.2f}x {'(JAX faster)' if speedup > 1 else '(PyTorch faster)'}")

def main():
    """Main test function."""
    print("JAX vs PyTorch Installation and Performance Test")
    print("=" * 60)
    
    # Check installations
    install_results = check_installations()
    
    # Only proceed with tests if both frameworks are installed
    if not (install_results.get('jax', {}).get('installed', False) and 
            install_results.get('torch', {}).get('installed', False)):
        print("\n❌ Cannot run performance tests - both JAX and PyTorch must be installed")
        sys.exit(1)
    
    # Run performance tests
    jax_times = test_jax_operations()
    torch_times = test_torch_operations()
    
    # Compare performance
    compare_performance(jax_times, torch_times)
    
    # Test gradient computation
    test_gradient_computation()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    # Summary
    jax_gpu = install_results.get('jax', {}).get('gpu_available', False)
    torch_gpu = install_results.get('torch', {}).get('cuda_available', False) or \
                install_results.get('torch', {}).get('mps_available', False)
    
    print(f"\nSummary:")
    print(f"- JAX GPU support: {'✓' if jax_gpu else '✗'}")
    print(f"- PyTorch GPU support: {'✓' if torch_gpu else '✗'}")
    
    if jax_gpu and torch_gpu:
        print("🎉 Both frameworks have GPU acceleration!")
    elif jax_gpu or torch_gpu:
        print("⚠️  Only one framework has GPU acceleration")
    else:
        print("🔄 Both frameworks running on CPU")

if __name__ == "__main__":
    main()
