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
 * \file moe_gating_top_k_softmax_tiling.h
 * \brief Tiling data structure for NpuMoeGatingTopKSoftmax
 */

#ifndef MOE_GATING_TOP_K_SOFTMAX_TILING_H
#define MOE_GATING_TOP_K_SOFTMAX_TILING_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, rowCount);
TILING_DATA_FIELD_DEF(int64_t, perCoreRowCount);
TILING_DATA_FIELD_DEF(int64_t, lastCoreRowCount);
TILING_DATA_FIELD_DEF(int64_t, expertCount);
TILING_DATA_FIELD_DEF(int64_t, addBias);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(float, routedScalingFactor);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(int64_t, calTmpBufUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax, MoeGatingTopKSoftmaxTilingData)
} // namespace optiling
#endif // MOE_GATING_TOP_K_SOFTMAX_TILING_H