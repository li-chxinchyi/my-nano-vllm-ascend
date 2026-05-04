/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * AddRmsNormBias tiling strategy implementation.
 * Determines how to partition work across AI cores and UB memory.
 */

#include "add_rms_norm_bias_tiling.h"
#include "log/ops_log.h"

namespace optiling {
constexpr uint32_t DTYPE_KEY_FP16 = 10;
constexpr uint32_t DTYPE_KEY_FP32 = 20;
constexpr uint32_t DTYPE_KEY_BF16 = 30;
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t UB_FACTOR_FP16 = 12288;
constexpr uint32_t UB_FACTOR_FP32 = 10240;
constexpr uint32_t BLOCK_ALIGN_NUM_FP16 = 16;
constexpr uint32_t BLOCK_ALIGN_NUM_FP32 = 8;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t BUFFER_NUM_VAL = 1;
constexpr uint32_t NUM_PER_REP_FP32_VAL = 64;
constexpr int32_t INPUT_X1_INDEX = 0;
constexpr int32_t INPUT_X2_INDEX = 1;
constexpr int32_t INPUT_GAMMA_INDEX = 2;
constexpr int32_t INPUT_BETA_INDEX = 3;
constexpr int32_t OUTPUT_Y_INDEX = 0;
constexpr int32_t OUTPUT_RSTD_INDEX = 1;
constexpr int32_t OUTPUT_X_INDEX = 2;

static void SetByDtype(ge::DataType dataType, uint32_t& dtypeKey, uint32_t& dataPerBlock)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            dataPerBlock = BLOCK_ALIGN_NUM_FP16;
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            dataPerBlock = BLOCK_ALIGN_NUM_FP16;
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;
            break;
    }
}

static bool CheckInputOutputShape(const gert::TilingContext* context)
{
    const gert::StorageShape* x1_shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape* x2_shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape* y_shape = context->GetOutputShape(OUTPUT_Y_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    size_t x1DimNum = x1_shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(gammaDimNum != 1, OP_LOGE(context, "Input gamma dimension must be 1"), return false);
    OP_CHECK_IF(x1DimNum < 1, OP_LOGE(context, "Input x1 dimension must be at least 1"), return false);

    int64_t hiddenSize = gamma_shape->GetStorageShape().GetDim(0);
    OP_CHECK_IF(hiddenSize <= 0, OP_LOGE(context, "Input gamma size must be positive"), return false);
    OP_CHECK_IF(x1_shape->GetStorageShape().GetDim(x1DimNum - 1) != hiddenSize,
        OP_LOGE(context, "Input x1 last dimension must match gamma size"), return false);

    return true;
}

static void GetCompileParameters(gert::TilingContext* context, uint32_t& numCore, uint64_t& ubSize)
{
    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormBiasCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
    }
    ubSize -= UB_USED;
}

static void CalculateRowAndColParameters(gert::TilingContext* context, uint32_t& numRow, uint32_t& numCol)
{
    const gert::Shape x1_shape = context->GetInputShape(INPUT_X1_INDEX)->GetStorageShape();
    numCol = context->GetInputShape(INPUT_GAMMA_INDEX)->GetStorageShape().GetDim(0);

    const size_t x1DimNum = x1_shape.GetDimNum();
    numRow = 1;
    for (size_t i = 0; i < x1DimNum - 1; ++i) {
        numRow *= x1_shape.GetDim(i);
    }
}

static ge::graphStatus GetEpsilonParameter(gert::TilingContext* context, float& epsilon)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    epsilon = *attrs->GetFloat(0);
    OP_CHECK_IF(epsilon < 0, OP_LOGE(context, "Epsilon must be non-negative"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void CalculateBlockParameters(uint32_t numRow, uint32_t numCore,
    uint32_t& blockFactor, uint32_t& lastBlockFactor, uint32_t& useCoreNum)
{
    blockFactor = 1;
    uint32_t tileNum = (numRow + numCore * blockFactor - 1) / (numCore * blockFactor);
    blockFactor *= tileNum;
    useCoreNum = (numRow + blockFactor - 1) / blockFactor;
    lastBlockFactor = numRow - blockFactor * (useCoreNum - 1);
}

static ge::graphStatus Tiling4AddRmsNormBias(gert::TilingContext* context)
{
    OPS_LOG_D(context, "Tiling4AddRmsNormBias running.");
    OP_CHECK_IF(!CheckInputOutputShape(context), OP_LOGE(context, "Input shape invalid"), return ge::GRAPH_FAILED);

    AddRmsNormBiasTilingData tiling;

    auto betaDesc = context->GetOptionalInputDesc(INPUT_BETA_INDEX);
    tiling.set_nullptr_beta(betaDesc == nullptr ? 1 : 0);

    uint32_t num_core;
    uint64_t ub_size;
    GetCompileParameters(context, num_core, ub_size);

    uint32_t num_row, num_col;
    CalculateRowAndColParameters(context, num_row, num_col);

    float epsilon = 0;
    ge::graphStatus ret = GetEpsilonParameter(context, epsilon);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    uint32_t block_factor, lastBlockFactor, use_core_num;
    CalculateBlockParameters(num_row, num_core, block_factor, lastBlockFactor, use_core_num);
    context->SetBlockDim(use_core_num);

    uint32_t dtype_key, data_per_block;
    auto data_type = context->GetInputDesc(0)->GetDataType();
    SetByDtype(data_type, dtype_key, data_per_block);

    uint32_t row_factor = 64;
    uint32_t ub_factor = (dtype_key == DTYPE_KEY_FP32) ? UB_FACTOR_FP32 : UB_FACTOR_FP16;
    uint32_t numColAlign = ((num_col + data_per_block - 1) / data_per_block) * data_per_block;

    uint32_t rowLoop = (block_factor + row_factor - 1) / row_factor;
    uint32_t lastBlockRowLoop = (lastBlockFactor + row_factor - 1) / row_factor;
    uint32_t rowTail = block_factor - (rowLoop - 1) * row_factor;
    uint32_t lastBlockRowTail = lastBlockFactor - (lastBlockRowLoop - 1) * row_factor;

    const float avgFactor = (num_col == 0) ? 0 : 1.0f / num_col;
    tiling.set_num_row(num_row);
    tiling.set_num_col(num_col);
    tiling.set_num_col_align(numColAlign);
    tiling.set_block_factor(block_factor);
    tiling.set_last_block_factor(lastBlockFactor);
    tiling.set_row_factor(row_factor);
    tiling.set_ub_factor(ub_factor);
    tiling.set_epsilon(epsilon);
    tiling.set_avg_factor(avgFactor);
    tiling.set_row_loop(rowLoop);
    tiling.set_last_block_row_loop(lastBlockRowLoop);
    tiling.set_row_tail(rowTail);
    tiling.set_last_block_row_tail(lastBlockRowTail);

    context->SetTilingKey(dtype_key);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaceSizes = context->GetWorkspaceSizes(0);
    *workspaceSizes = 0;

    OPS_LOG_I(context, "Tiling Key: %u, Block Dim: %u", dtype_key, use_core_num);
    OPS_LOG_I(context,
        "num_row: %u, num_col: %u, block_factor: %u, row_factor: %u, ub_factor: %u, epsilon: %f",
        num_row, num_col, block_factor, row_factor, ub_factor, epsilon);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AddRmsNormBias(gert::TilingParseContext* context)
{
    OPS_LOG_D(context, "TilingPrepare4AddRmsNormBias running.");
    auto compileInfo = context->GetCompiledInfo<AddRmsNormBiasCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddRmsNormBias).Tiling(Tiling4AddRmsNormBias).TilingParse<AddRmsNormBiasCompileInfo>(TilingPrepare4AddRmsNormBias);
}
