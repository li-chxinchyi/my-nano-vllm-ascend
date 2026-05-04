/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * RMS Norm base utilities for Ascend C kernel.
 * Provides common constants, helper functions, and reduce operations.
 */

#ifndef RMS_NORM_BASE_H_
#define RMS_NORM_BASE_H_
#include "kernel_operator.h"
#include "reduce_common.h"

namespace RmsNorm {
using namespace AscendC;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220 && __CCE_AICORE__ != 310
#define bfloat16_t int16_t
#endif

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t NUM_PER_REP_FP32 = 64;
constexpr int32_t NUM_PER_BLK_FP32 = 8;
constexpr int32_t BLOCK_SIZE = 32;
constexpr uint32_t ONE_REPEAT_BYTE_SIZE_VAL = 256;
constexpr float ONE = 1;
constexpr float ZERO_VAL = 0;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

template <typename T>
__aicore__ inline T Min(T left, T right)
{
    return (left < right ? left : right);
}

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U& dstTensor, const R& srcTensor, const uint32_t count)
{
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    DataCopyParams copyParams;
    copyParams.blockLen = count * sizeof(T);
    copyParams.blockCount = 1;
    if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
        DataCopyPadParams padParams;
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    } else {
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }
#else
    int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
            int32_t num = AlignUp(count, numPerBlock);
            DataCopy(dstTensor, srcTensor, num);
        } else {
            if (count < numPerBlock) {
                DataCopy(dstTensor, srcTensor, numPerBlock);
            } else {
                int32_t num = count / numPerBlock * numPerBlock;
                DataCopy(dstTensor, srcTensor, num);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (int32_t i = 0; i < numPerBlock; i++) {
                    T tensorValue = srcTensor.GetValue(count - numPerBlock + i);
                    srcTensor.SetValue(i, tensorValue);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
            }
        }
    }
#endif
}

__aicore__ inline void ReduceSumFP32(
    const LocalTensor<float>& dst_local,
    const LocalTensor<float>& src_local,
    const LocalTensor<float>& work_local,
    int32_t count)
{
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    if (likely(repeatTimes > 0)) {
        Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dst_local, work_local, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#elif !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    WholeReduceSum<float, false>(dst_local, work_local, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumCustom(
    const LocalTensor<float>& dst_local,
    const LocalTensor<float>& src_local,
    const LocalTensor<float>& work_local,
    int32_t count)
{
    ReduceSumFP32(dst_local, src_local, work_local, count);
}

} // namespace RmsNorm
#endif // RMS_NORM_BASE_H_
