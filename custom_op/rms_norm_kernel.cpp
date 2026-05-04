/**
 * Ascend C RMS Norm Kernel
 *
 * 真正运行在 AI Core 上的 RMS Norm 实现。
 * 所有计算在 UB (Unified Buffer) 内完成，数据不反复读写 Global Memory。
 *
 * 公式: output[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t NUM_PER_REP_FP32 = 64;  // 256 bytes / 4 bytes

template <typename T>
__aicore__ inline uint32_t AlignUp(uint32_t n) {
    uint32_t perBlock = ONE_BLK_SIZE / sizeof(T);
    return ((n + perBlock - 1) / perBlock) * perBlock;
}

class KernelRmsNorm {
public:
    __aicore__ inline KernelRmsNorm() {}

    __aicore__ inline void Init(
        __gm__ void* x_gm,
        __gm__ void* gamma_gm,
        __gm__ void* y_gm,
        uint32_t num_row,
        uint32_t num_col,
        float epsilon)
    {
        this->numRow = num_row;
        this->numCol = num_col;
        this->epsilon = epsilon;
        this->avgFactor = (num_col != 0) ? 1.0f / num_col : 0.0f;

        uint32_t aivNum = GetBlockNum();
        uint32_t aivIdx = GetBlockIdx();

        this->rowPerCore = (numRow + aivNum - 1) / aivNum;
        this->startRow = aivIdx * rowPerCore;
        this->endRow = (startRow + rowPerCore > numRow) ? numRow : startRow + rowPerCore;
        if (startRow >= numRow) {
            this->endRow = startRow;
        }

        xGm.SetGlobalBuffer((__gm__ half*)x_gm + startRow * numCol);
        gammaGm.SetGlobalBuffer((__gm__ half*)gamma_gm);
        yGm.SetGlobalBuffer((__gm__ half*)y_gm + startRow * numCol);

        uint32_t colAlignHalf = AlignUp<half>(numCol);
        uint32_t colAlignFloat = AlignUp<float>(numCol);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, colAlignHalf * sizeof(half));
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, colAlignHalf * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, colAlignHalf * sizeof(half));
        pipe.InitBuffer(fp32Buf, colAlignFloat * sizeof(float));
        pipe.InitBuffer(sqBuf, colAlignFloat * sizeof(float));
        pipe.InitBuffer(reduceBuf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (startRow >= endRow) return;

        uint32_t copyLen = AlignUp<half>(numCol);

        // gamma -> UB
        LocalTensor<half> gammaLocal = inQueueGamma.AllocTensor<half>();
        DataCopy(gammaLocal, gammaGm, copyLen);
        inQueueGamma.EnQue(gammaLocal);
        gammaLocal = inQueueGamma.DeQue<half>();

        for (uint32_t row = 0; row < endRow - startRow; row++) {
            uint32_t offset = row * numCol;

            // ---- CopyIn ----
            LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
            DataCopy(xLocal, xGm[offset], copyLen);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<half>();

            // ---- Compute ----
            LocalTensor<float> xFp32 = fp32Buf.Get<float>();
            LocalTensor<float> sqx = sqBuf.Get<float>();
            LocalTensor<float> reduceWork = reduceBuf.Get<float>();

            Cast(xFp32, xLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            Mul(sqx, xFp32, xFp32, numCol);
            PipeBarrier<PIPE_V>();

            Muls(sqx, sqx, avgFactor, numCol);
            PipeBarrier<PIPE_V>();

            // reduce sum
            uint64_t mask = NUM_PER_REP_FP32;
            int32_t repeatTimes = numCol / NUM_PER_REP_FP32;
            int32_t tailCount = numCol % NUM_PER_REP_FP32;
            int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;

            BinaryRepeatParams rp;
            rp.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
            rp.src0BlkStride = 1;
            rp.src1RepStride = 0;
            rp.src1BlkStride = 1;
            rp.dstRepStride = 0;
            rp.dstBlkStride = 1;

            Duplicate(reduceWork, 0.0f, NUM_PER_REP_FP32);
            PipeBarrier<PIPE_V>();
            if (repeatTimes > 0) {
                Add(reduceWork, sqx, reduceWork, mask, repeatTimes, rp);
                PipeBarrier<PIPE_V>();
            }
            if (tailCount != 0) {
                Add(reduceWork, sqx[bodyCount], reduceWork, tailCount, 1, rp);
                PipeBarrier<PIPE_V>();
            }

            AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            if (g_coreType == AIV) {
                WholeReduceSum<float, false>(sqx, reduceWork, MASK_PLACEHOLDER, 1, 0, 1, 0);
            }
#else
            WholeReduceSum<float, false>(sqx, reduceWork, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
            PipeBarrier<PIPE_V>();

            Adds(sqx, sqx, epsilon, 1);
            PipeBarrier<PIPE_V>();

            Sqrt(sqx, sqx, 1);
            Duplicate(reduceWork, 1.0f, 1);
            PipeBarrier<PIPE_V>();
            Div(sqx, reduceWork, sqx, 1);
            PipeBarrier<PIPE_V>();

            event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evVS);
            WaitFlag<HardEvent::V_S>(evVS);
            float rstdVal = sqx.GetValue(0);
            event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evSV);
            WaitFlag<HardEvent::S_V>(evSV);

            Muls(xFp32, xFp32, rstdVal, numCol);
            PipeBarrier<PIPE_V>();

            LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
            Cast(yLocal, xFp32, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            Mul(yLocal, gammaLocal, yLocal, numCol);
            PipeBarrier<PIPE_V>();

            inQueueX.FreeTensor(xLocal);

            // ---- CopyOut ----
            outQueueY.EnQue(yLocal);
            yLocal = outQueueY.DeQue<half>();
            DataCopy(yGm[offset], yLocal, copyLen);
            outQueueY.FreeTensor(yLocal);
        }

        inQueueGamma.FreeTensor(gammaLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> fp32Buf;
    TBuf<TPosition::VECCALC> sqBuf;
    TBuf<TPosition::VECCALC> reduceBuf;

    GlobalTensor<half> xGm;
    GlobalTensor<half> gammaGm;
    GlobalTensor<half> yGm;

    uint32_t numRow;
    uint32_t numCol;
    float epsilon;
    float avgFactor;
    uint32_t rowPerCore;
    uint32_t startRow;
    uint32_t endRow;
};

extern "C" __global__ __aicore__ void rms_norm_ascendc_kernel(
    __gm__ void* x,
    __gm__ void* gamma,
    __gm__ void* y,
    uint32_t num_row,
    uint32_t num_col,
    float epsilon)
{
    KernelRmsNorm op;
    op.Init(x, gamma, y, num_row, num_col, epsilon);
    op.Process();
}

namespace custom_op {
void rms_norm_ascendc_impl(
    void* stream, void* x, void* gamma, void* y,
    uint32_t num_row, uint32_t num_col, float epsilon, uint32_t aiv_num)
{
    rms_norm_ascendc_kernel<<<aiv_num, nullptr, stream>>>(
        x, gamma, y, num_row, num_col, epsilon);
}
}
