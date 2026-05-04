/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file moe_gating_top_k_softmax.h
 * \brief Kernel header for NpuMoeGatingTopKSoftmax
 */

#ifndef MOE_GATING_TOP_K_SOFTMAX_H
#define MOE_GATING_TOP_K_SOFTMAX_H

#include "kernel_operator.h"
#include <cmath>

namespace MoeGatingTopKSoftmax {
using namespace AscendC;

constexpr int32_t FLOAT32_NEG_INF = 0xFF800000; // -inf -2139095040
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t REPEAT_BYTES = 256;
constexpr int64_t REPEAT_BLOCKS = 8;

constexpr int32_t CONSTANT_TWO = 2;
constexpr int32_t CONSTANT_THREE = 3;
constexpr int32_t CONSTANT_FOUR = 2;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 x, T2 y)
{
    if (y != 0 && x != 0) {
        const T1 quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }
    return x;
}

template <typename T>
class MoeGatingTopKSoftmaxKernel {
public:
    __aicore__ inline MoeGatingTopKSoftmaxKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out,
                                GM_ADDR workspace, const MoeGatingTopKSoftmaxTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBiasAndInitExpertId();
    __aicore__ inline void CopyInX(int64_t row);
    __aicore__ inline void ComputeSoftmaxCopy();
    __aicore__ inline void CopyOutXNorm(int64_t row);
    __aicore__ inline void SortAll();
    __aicore__ inline void SelectTopK();
    __aicore__ inline void SelectTopKExpertScore();
    __aicore__ inline void CumputeActualTopKExpertId();
    __aicore__ inline void CopyOut(int64_t row);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
    TQue<QuePosition::VECOUT, 1> outOutQueue_;

    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> expertIdBuf_;
    TBuf<TPosition::VECCALC> xNormBuf_;
    TBuf<TPosition::VECCALC> sortedBuf_;
    TBuf<TPosition::VECCALC> topKExpertIdBuf_;
    TBuf<TPosition::VECCALC> calcTmpBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<float> outGm_;

    int64_t blockIdx_ = 0;
    int64_t perCoreRowCount_ = 0;
    int64_t curCoreRowCount_ = 0;
    int64_t expertCount_ = 0;
    int64_t expertCountAlign_ = 0;
    bool addBias_ = false;
    int64_t k_ = 0;
    int64_t kAlign_ = 0;

    const MoeGatingTopKSoftmaxTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::CopyInBiasAndInitExpertId()
{
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();
    LocalTensor<int32_t> expertIdTensor = expertIdBuf_.Get<int32_t>();

    if (addBias_) {
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(biasTensor, biasGm_);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        } else {
            DataCopy(biasTensor[expertCountAlign_].ReinterpretCast<T>(), biasGm_);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(biasTensor, biasTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE,
                 expertCountAlign_);
            PipeBarrier<PIPE_V>();
        }
    }
    ArithProgression(expertIdTensor, static_cast<int32_t>(0), static_cast<int32_t>(1), expertCountAlign_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::CopyInX(int64_t row)
{
    LocalTensor<float> xInLocalTensor = xInQueue_.AllocTensor<float>();

    if constexpr (IsSameType<T, float>::value) {
        DataCopy(xInLocalTensor, xGm_[row * expertCount_], expertCount_);
    } else {
        DataCopy(xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), xGm_[row * expertCount_], expertCount_);
    }
    xInQueue_.EnQue(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::ComputeSoftmaxCopy()
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xInLocalTensor = xInQueue_.DeQue<float>();
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();

    if constexpr (!IsSameType<T, float>::value) {
        Cast(xInLocalTensor, xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE,
             expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }

    int64_t duplicateNum = expertCount_ % ONE_REPEAT_SORT_NUM;
    int duplicateIndex = expertCount_ - duplicateNum;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(xInLocalTensor.ReinterpretCast<int32_t>()[duplicateIndex], FLOAT32_NEG_INF, mask, 1, 1,
                  (expertCountAlign_ * sizeof(float)) / BLOCK_BYTES);
        PipeBarrier<PIPE_V>();
    }

    // Softmax computation
    LocalTensor<float> reduceValueTensor = calcTmpBuf_.Get<float>();
    LocalTensor<float> calcTmp = calcTmpBuf_.Get<float>()[BLOCK_BYTES];

    // Compute max
    ReduceMax(reduceValueTensor, xInLocalTensor, calcTmp, expertCountAlign_);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float maxValue = reduceValueTensor.GetValue(0);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    // Subtract max and exp
    Adds(xNormTensor, xInLocalTensor, -maxValue, expertCountAlign_);
    PipeBarrier<PIPE_V>();
    Exp(xNormTensor, xNormTensor, expertCountAlign_);
    PipeBarrier<PIPE_V>();

    // Sum for normalization
    ReduceSum(reduceValueTensor, xNormTensor, calcTmp, expertCountAlign_);
    eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float sumValue = reduceValueTensor.GetValue(0) + tilingData_->eps;
    eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    // Normalize
    Muls(xNormTensor, xNormTensor, 1.0f / sumValue, expertCountAlign_);
    PipeBarrier<PIPE_V>();

    if (addBias_) {
        Adds(xNormTensor, xNormTensor, biasTensor, expertCountAlign_);
    }

    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        PipeBarrier<PIPE_V>();
        Duplicate(xNormTensor.ReinterpretCast<int32_t>()[duplicateIndex],
                  FLOAT32_NEG_INF,
                  mask, 1, 1, (expertCountAlign_ * sizeof(float)) / BLOCK_BYTES);
    }
    xInQueue_.FreeTensor(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::CopyOutXNorm(int64_t row)
{
    LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    DataCopy(outOutTensor, xNormTensor, expertCountAlign_);
    outOutQueue_.EnQue<float>(outOutTensor);
    outOutTensor = outOutQueue_.DeQue<float>();
    DataCopy(outGm_[row * expertCount_], outOutTensor, expertCount_);
    outOutQueue_.FreeTensor(outOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::SortAll()
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<uint32_t> expertIdTensor = expertIdBuf_.Get<uint32_t>();
    LocalTensor<float> sortedTensor = sortedBuf_.Get<float>();
    LocalTensor<float> tmpLocal = calcTmpBuf_.Get<float>();

    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedTensor, xNormTensor, expertIdTensor, tmpLocal,
                      CeilDiv(expertCountAlign_, ONE_REPEAT_SORT_NUM));
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::SelectTopK()
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<uint32_t> expertIdTensor = expertIdBuf_.Get<uint32_t>();
    LocalTensor<float> sortedTensor = sortedBuf_.Get<float>();
    LocalTensor<int32_t> topKExpertId = topKExpertIdBuf_.Get<int32_t>();
    LocalTensor<float> mrgSort0Tensor = calcTmpBuf_.Get<float>();

    PipeBarrier<PIPE_V>();
    // Use TopK-like behavior: extract top k from sorted results
    uint8_t src1Pattern = 2; // Extract indices
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 0;
    gatherMaskParams.src1RepeatStride = 0;

    uint64_t rsvdCnt = 0;
    PipeBarrier<PIPE_V>();
    GatherMask(topKExpertId, sortedTensor.template ReinterpretCast<int32_t>(),
               src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::SelectTopKExpertScore()
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> yOutTensor = yOutQueue_.AllocTensor<float>();
    LocalTensor<int32_t> topKExpertId = topKExpertIdBuf_.Get<int32_t>();
    LocalTensor<int32_t> topKExpertIdWithByte = calcTmpBuf_.Get<int32_t>();

    // Get top k values using sorted indices
    PipeBarrier<PIPE_V>();
    Muls(topKExpertIdWithByte, topKExpertId, static_cast<int32_t>(sizeof(float)), k_);
    PipeBarrier<PIPE_V>();

    Gather(yOutTensor, xNormTensor, topKExpertIdWithByte.template ReinterpretCast<uint32_t>(),
           static_cast<uint32_t>(0), k_);
    PipeBarrier<PIPE_V>();

    // Apply scaling factor
    Muls(yOutTensor, yOutTensor, tilingData_->routedScalingFactor, k_);
    PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        Cast(yOutTensor.ReinterpretCast<T>(), yOutTensor, RoundMode::CAST_RINT, k_);
    }

    yOutQueue_.EnQue<float>(yOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::CumputeActualTopKExpertId()
{
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> topKExpertId = topKExpertIdBuf_.Get<int32_t>();
    LocalTensor<float> topKExpertIdFp32 = calcTmpBuf_.Get<float>();

    PipeBarrier<PIPE_V>();
    Cast(topKExpertIdFp32, topKExpertId, RoundMode::CAST_ROUND, k_);
    PipeBarrier<PIPE_V>();
    Muls(topKExpertIdFp32, topKExpertIdFp32, 1.0f / (float)expertCountAlign_, k_);
    PipeBarrier<PIPE_V>();
    Cast(expertIdxOut, topKExpertIdFp32, RoundMode::CAST_TRUNC, k_);
    PipeBarrier<PIPE_V>();
    Muls(expertIdxOut, expertIdxOut, static_cast<int32_t>(expertCountAlign_ - expertCount_), k_);
    PipeBarrier<PIPE_V>();
    Sub(expertIdxOut, topKExpertId, expertIdxOut, k_);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::CopyOut(int64_t row)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[row * k_], yOutTensor, dataCopyParams);
    dataCopyParams.blockLen = k_ * sizeof(int32_t);
    DataCopyPad(expertIdxGm_[row * k_], expertIdxOut, dataCopyParams);
    yOutQueue_.FreeTensor(yOutTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxOut);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                         GM_ADDR out, GM_ADDR workspace,
                                                         const MoeGatingTopKSoftmaxTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    addBias_ = tilingData_->addBias == 1;
    k_ = tilingData_->k;

    expertCountAlign_ = Align(expertCount_, sizeof(float));
    kAlign_ = Align(k_, sizeof(float));

    // init input gm buf
    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);

    // init output gm buf
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ float *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    // init que
    pipe_->InitBuffer(xInQueue_, 1, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(yOutQueue_, 1, kAlign_ * sizeof(float));
    pipe_->InitBuffer(expertIdxOutQueue_, 1, kAlign_ * sizeof(int32_t));
    pipe_->InitBuffer(outOutQueue_, 1, expertCountAlign_ * sizeof(float));

    pipe_->InitBuffer(biasBuf_, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(expertIdBuf_, expertCountAlign_ * sizeof(int32_t));

    pipe_->InitBuffer(xNormBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(sortedBuf_, expertCountAlign_ * (sizeof(float) + sizeof(uint32_t)));
    pipe_->InitBuffer(topKExpertIdBuf_, kAlign_ * sizeof(int32_t));
    pipe_->InitBuffer(calcTmpBuf_, expertCountAlign_ * sizeof(float) * 10);
}

template <typename T>
__aicore__ inline void MoeGatingTopKSoftmaxKernel<T>::Process()
{
    CopyInBiasAndInitExpertId();
    for (int64_t row = 0; row < curCoreRowCount_; row++) {
        CopyInX(row);
        ComputeSoftmaxCopy();
        CopyOutXNorm(row);
        SortAll();
        SelectTopK();
        SelectTopKExpertScore();
        CumputeActualTopKExpertId();
        CopyOut(row);
    }
}
} // namespace MoeGatingTopKSoftmax
#endif // MOE_GATING_TOP_K_SOFTMAX_H