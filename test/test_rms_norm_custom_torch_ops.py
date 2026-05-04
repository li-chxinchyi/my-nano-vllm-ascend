#!/usr/bin/env python3
"""
Test script for RMS Norm custom operator using torch.ops._C_ascend
"""
import torch
import torch_npu
from nanovllm.layers.rms_norm_custom import rms_norm, rms_norm_with_rstd, RMSNormCustom, _has_custom_op

def test_rms_norm_basic():
    print("\n=== Testing RMS Norm Basic ===")
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Custom operator available: {_has_custom_op}")

    hidden_size = 4096
    batch_size = 8

    input_tensor = torch.randn(batch_size, hidden_size, device=device)
    weight = torch.ones(hidden_size, device=device)

    output = rms_norm(input_tensor, weight, epsilon=1e-6)
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")

    # Verify correctness against Python implementation
    x = input_tensor.cpu()
    orig_dtype = x.dtype
    x_float = x.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = x_float.mul_(torch.rsqrt(var + 1e-6))
    expected_output = normalized.to(orig_dtype).mul_(weight.cpu())

    error = (output.cpu() - expected_output).abs().max()
    print(f"Max error vs Python implementation: {error.item():.6e}")
    print(f"Check passed: {error < 1e-3}")


def test_rms_norm_with_rstd():
    print("\n=== Testing RMS Norm with RSTD ===")
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    print(f"Device: {device}")

    hidden_size = 4096
    batch_size = 8

    input_tensor = torch.randn(batch_size, hidden_size, device=device)
    weight = torch.ones(hidden_size, device=device)

    output, rstd = rms_norm_with_rstd(input_tensor, weight, epsilon=1e-6)
    print(f"Output shape: {output.shape}")
    print(f"RSTD shape: {rstd.shape}")
    print(f"RSTD mean: {rstd.mean().item():.4f}")
    print(f"RSTD std: {rstd.std().item():.4f}")

    # Verify correctness
    x = input_tensor.cpu()
    x_float = x.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    expected_rstd = torch.rsqrt(var + 1e-6).squeeze(-1)
    rstd_error = (rstd.cpu() - expected_rstd).abs().max()
    print(f"Max RSTD error vs Python implementation: {rstd_error.item():.6e}")


def test_rms_norm_layer():
    print("\n=== Testing RMS Norm Layer ===")
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")

    hidden_size = 4096
    batch_size = 8

    norm_layer = RMSNormCustom(hidden_size, eps=1e-6).to(device)
    input_tensor = torch.randn(batch_size, hidden_size, device=device)

    output = norm_layer(input_tensor)
    print(f"Layer output shape: {output.shape}")
    print(f"Layer output mean: {output.mean().item():.4f}")


def test_torch_ops_interface():
    print("\n=== Testing torch.ops._C_ascend Interface ===")
    if _has_custom_op:
        device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
        hidden_size = 4096
        batch_size = 8

        input_tensor = torch.randn(batch_size, hidden_size, device=device)
        weight = torch.ones(hidden_size, device=device)

        # Test direct torch.ops._C_ascend call
        output = torch.ops._C_ascend.rms_norm_custom(input_tensor, weight, 1e-6)
        print(f"Direct call via torch.ops._C_ascend.rms_norm_custom: {output.shape}")

        # Test with rstd
        output, rstd = torch.ops._C_ascend.rms_norm_custom_with_rstd(input_tensor, weight, 1e-6)
        print(f"Direct call via torch.ops._C_ascend.rms_norm_custom_with_rstd: {output.shape}, {rstd.shape}")
    else:
        print("Custom operator not available, skipping direct torch.ops._C_ascend test")


if __name__ == "__main__":
    print("Testing RMS Norm Custom Operator (torch.ops._C_ascend)")
    print("=" * 60)

    test_rms_norm_basic()
    test_rms_norm_with_rstd()
    test_rms_norm_layer()
    test_torch_ops_interface()

    print("\n" + "=" * 60)
    print("All tests completed")