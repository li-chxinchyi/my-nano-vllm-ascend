/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * AddRmsNormBias tiling data structure definition.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_H_
#include "register/tilingdata_base.h"
#include "log/ops_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AddRmsNormBiasTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_row)
TILING_DATA_FIELD_DEF(uint32_t, num_col)
TILING_DATA_FIELD_DEF(uint32_t, block_factor)
TILING_DATA_FIELD_DEF(uint32_t, row_factor)
TILING_DATA_FIELD_DEF(uint32_t, ub_factor)
TILING_DATA_FIELD_DEF(float, epsilon)
TILING_DATA_FIELD_DEF(float, avg_factor)
TILING_DATA_FIELD_DEF(uint32_t, num_col_align)
TILING_DATA_FIELD_DEF(uint32_t, last_block_factor)
TILING_DATA_FIELD_DEF(uint32_t, row_loop)
TILING_DATA_FIELD_DEF(uint32_t, last_block_row_loop)
TILING_DATA_FIELD_DEF(uint32_t, row_tail)
TILING_DATA_FIELD_DEF(uint32_t, last_block_row_tail)
TILING_DATA_FIELD_DEF(uint32_t, nullptr_beta)
END_TILING_DATA_DEF

struct AddRmsNormBiasCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910_95;
};

REGISTER_TILING_DATA_CLASS(AddRmsNormBias, AddRmsNormBiasTilingData)
}
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_H_
