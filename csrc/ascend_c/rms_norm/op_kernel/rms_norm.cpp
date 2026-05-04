/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc.h"
#include "rms_norm.h"

using namespace AscendC;

struct ReduceFp32Buf {
    float data[64];
};

#define GENERAL_OP_IMPL(templateClass, ...)                                              \
    do {                                                                                 \
        templateClass<__VA_ARGS__> op(&pipe);                                           \
        op.Init(x, gamma, y, &tilingData);                                              \
        op.Process();                                                                   \
    } while (0)

extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(KernelRmsNorm, half);
    } else if (TILING_KEY_IS(20)) {
        GENERAL_OP_IMPL(KernelRmsNorm, float);
    } else if (TILING_KEY_IS(30)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelRmsNorm, bfloat16_t);
#endif
    }
}