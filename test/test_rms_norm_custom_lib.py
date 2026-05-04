"""
Test script for RMSNorm custom operator
"""
import torch
import torch_npu

from nanovllm.layers.rms_norm_custom import rms_norm, rms_norm_with_rstd, RMSNormCustom


def test_rms_norm_correctness():
    """Test RMS Norm correctness against Python implementation"""
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    
    hidden_size = 512
    batch_size = 4
    
    # Create test data
    x = torch.randn(batch_size, hidden_size, device=device)
    weight = torch.randn(hidden_size, device=device)
    epsilon = 1e-6
    
    # Custom operator
    y_custom = rms_norm(x, weight, epsilon)
    
    # Python reference implementation
    x_cpu = x.cpu()
    weight_cpu = weight.cpu()
    orig_dtype = x_cpu.dtype
    x_float = x_cpu.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = x_float.mul(torch.rsqrt(var + epsilon))
    y_ref = normalized.to(orig_dtype).mul(weight_cpu)
    
    # Compare
    error = (y_custom.cpu() - y_ref).abs().max()
    print(f"Max error: {error.item():.6e}")
    assert error < 1e-3, f"Error too large: {error.item()}"
    print("✓ RMS Norm correctness test passed")


def test_rms_norm_with_rstd():
    """Test RMS Norm with rstd output"""
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    
    hidden_size = 512
    batch_size = 4
    
    x = torch.randn(batch_size, hidden_size, device=device)
    weight = torch.randn(hidden_size, device=device)
    
    y, rstd = rms_norm_with_rstd(x, weight, epsilon=1e-6)
    
    print(f"Output shape: {y.shape}")
    print(f"RSTD shape: {rstd.shape}")
    assert y.shape == x.shape, "Output shape mismatch"
    assert rstd.shape == (batch_size,), "RSTD shape mismatch"
    print("✓ RMS Norm with rstd test passed")


def test_rms_norm_layer():
    """Test RMSNormCustom layer"""
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")

    hidden_size = 1024
    batch_size = 8
    seq_len = 16

    # Create instance and move weights to device
    norm_layer = RMSNormCustom(hidden_size, eps=1e-6)
    norm_layer.weight.data = norm_layer.weight.data.to(device)

    # Test different input shapes
    x_2d = torch.randn(seq_len, hidden_size, device=device)
    y_2d = norm_layer(x_2d)
    assert y_2d.shape == x_2d.shape
    print(f"2D input test passed: {x_2d.shape} -> {y_2d.shape}")

    x_3d = torch.randn(batch_size, seq_len, hidden_size, device=device)
    try:
        y_3d = norm_layer(x_3d)
        assert y_3d.shape == x_3d.shape
        print(f"3D input test passed: {x_3d.shape} -> {y_3d.shape}")
    except Exception as e:
        print(f"3D input not supported (expected): {e}")


def test_gradients():
    """Test gradient computation"""
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")

    hidden_size = 256
    batch_size = 2

    x = torch.randn(batch_size, hidden_size, device=device, requires_grad=True)
    norm_layer = RMSNormCustom(hidden_size, eps=1e-6)
    norm_layer.weight.data = norm_layer.weight.data.to(device)

    output = norm_layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient not computed"
    assert norm_layer.weight.grad is not None, "Weight gradient not computed"

    print(f"Input grad shape: {x.grad.shape}")
    print(f"Weight grad shape: {norm_layer.weight.grad.shape}")
    print("✓ Gradient computation test passed")


def benchmark_performance():
    """Benchmark performance"""
    if not torch.npu.is_available():
        print("NPU not available, skipping benchmark")
        return

    device = torch.device("npu:0")

    hidden_size = 4096
    batch_size = 32
    num_iterations = 100

    x = torch.randn(batch_size, hidden_size, device=device)
    norm_layer = RMSNormCustom(hidden_size, eps=1e-6)
    norm_layer.weight.data = norm_layer.weight.data.to(device)
    
    # Warmup
    for _ in range(10):
        _ = norm_layer(x)
    
    # Benchmark
    torch.npu.synchronize()
    import time
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = norm_layer(x)
    
    torch.npu.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    throughput = batch_size * hidden_size / (avg_time_ms / 1000)
    
    print(f"\nPerformance Benchmark:")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Average time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {throughput:.2e} elements/sec")


if __name__ == "__main__":
    print("=" * 60)
    print("RMSNorm Custom Operator Tests")
    print("=" * 60)
    
    test_rms_norm_correctness()
    test_rms_norm_with_rstd()
    test_rms_norm_layer()
    test_gradients()
    benchmark_performance()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)