/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * AddRmsNormBias kernel entry point.
 * Dispatches to the correct template instantiation based on tiling key (dtype).
 */

#include "ascendc.h"
#include "add_rms_norm_bias.h"

using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)              \
    do {                                                 \
        templateClass<__VA_ARGS__> op(&pipe);            \
        op.Init(x1, x2, gamma, beta, y, rstd, x, &tilingData); \
        op.Process();                                    \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_bias(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBias, half);
    } else if (TILING_KEY_IS(20)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBias, float);
    } else if (TILING_KEY_IS(30)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBias, bfloat16_t);
#endif
    }
}
