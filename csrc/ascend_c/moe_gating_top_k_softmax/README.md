# NpuMoeGatingTopKSoftmax AscendC Operator

## Description
Custom AscendC implementation of npu_moe_gating_top_k_softmax operator.

## Inputs
- `x`: A 2D tensor with shape (rows, expert_num), dtype: float16/float/bfloat16
- `bias`: Optional 1D tensor with shape (expert_num), same dtype as x
- `k`: Number of top-k experts to select

## Outputs
- `y`: Tensor with shape (rows, k), top-k routing weights, same dtype as x
- `expert_idx`: Tensor with shape (rows, k), top-k expert indices, dtype: int32
- `out`: Tensor with shape (rows, expert_num), softmax output, dtype: float32