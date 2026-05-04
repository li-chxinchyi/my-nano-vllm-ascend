/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_RMS_NORM_IMPL_H_
#define KERNEL_RMS_NORM_IMPL_H_
#include "rms_norm.h"
extern "C" {
#include "op_host/rms_norm_tiling.h"
}
using namespace AscendC;
using namespace optiling;

template <typename T>
class KernelRmsNorm {
public:
    __aicore__ inline KernelRmsNorm(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
        const RmsNormTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        this->numRow = tiling->num_row;
        this->numCol = tiling->num_col;
        this->blockFactor = tiling->block_factor;
        this->rowFactor = tiling->row_factor;
        this->ubFactor = tiling->ub_factor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = (numCol != 0) ? (float) 1.0 / numCol : 0;

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * blockFactor * numCol, rowWork * numCol);

        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();

        uint32_t i_o_max = RmsNorm::CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;

        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, gammaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal);

        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T>& gammaLocal)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
            CopyIn(gm_bias);
            Compute(i_i, gammaLocal, rstdLocal);
            CopyOutY(gm_bias);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyCustom<T>(xLocal, xGm[gm_bias], numCol);
        inQueueX.EnQue(xLocal);
        auto xLocalIn = inQueueX.DeQue<T>();
        inQueueX.FreeTensor(xLocalIn);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute(
        uint32_t inner_progress,
        LocalTensor<T>& gammaLocal,
        LocalTensor<float> rstdLocal)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        Mul(sqx, xLocal, xLocal, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        PipeBarrier<PIPE_V>();

        Div(sqx, reduce_buf_local, sqx, 1);
        PipeBarrier<PIPE_V>();

        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);

        float rstdValue = sqx.GetValue(0);

        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        rstdLocal.SetValue(inner_progress, rstdValue);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Muls(yLocal, xLocal, rstdValue, numCol);
        inQueueX.FreeTensor(xLocal);
        PipeBarrier<PIPE_V>();

        Mul(yLocal, gammaLocal, yLocal, numCol);
        PipeBarrier<PIPE_V>();

        outQueueY.EnQue<float>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[progress], yLocal, numCol);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor;
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
};
#endif // KERNEL_RMS_NORM_IMPL_H_