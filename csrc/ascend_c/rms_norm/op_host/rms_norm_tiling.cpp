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
 * \file rms_norm_tiling.cpp
 * \brief RMS Norm tiling implementation
 */
#include "rms_norm_tiling.h"
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
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t NUM_PER_REP_FP32 = 64;
constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_GAMMA_INDEX = 1;
constexpr int32_t OUTPUT_Y_INDEX = 0;
constexpr int32_t TEN = 10;

platform_ascendc::SocVersion rmsNormSocVersion;

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
    const gert::StorageShape* x_shape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape* y_shape = context->GetOutputShape(OUTPUT_Y_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    size_t xDimNum = x_shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();
    size_t yDimNum = y_shape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(gammaDimNum != 1, OP_LOGE(context, "Input gamma dimension must be 1"), return false);
    OP_CHECK_IF(xDimNum < 1, OP_LOGE(context, "Input x dimension must be at least 1"), return false);
    OP_CHECK_IF(xDimNum != yDimNum, OP_LOGE(context, "Input x dimension must equal output y dimension"),
        return false);

    int64_t hiddenSize = gamma_shape->GetStorageShape().GetDim(0);
    OP_CHECK_IF(hiddenSize <= 0, OP_LOGE(context, "Input gamma size must be positive"), return false);
    OP_CHECK_IF(x_shape->GetStorageShape().GetDim(xDimNum - 1) != hiddenSize,
        OP_LOGE(context, "Input x last dimension must match gamma size"), return false);

    return true;
}

static void GetCompileParameters(gert::TilingContext* context, uint32_t& numCore, uint64_t& ubSize)
{
    auto ptrCompileInfo = reinterpret_cast<const RmsNormCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        rmsNormSocVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
        rmsNormSocVersion = ptrCompileInfo->socVersion;
    }
    ubSize -= UB_USED;
}

static void CalculateRowAndColParameters(gert::TilingContext* context, uint32_t& numRow, uint32_t& numCol)
{
    const gert::Shape x_shape = context->GetInputShape(0)->GetStorageShape();
    numCol = context->GetInputShape(1)->GetStorageShape().GetDim(0);

    const size_t x_dim_num = x_shape.GetDimNum();
    numRow = 1;
    for (size_t i = 0; i < x_dim_num - 1; ++i) {
        numRow *= x_shape.GetDim(i);
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

static ge::DataType SetDataTypeParameters(gert::TilingContext* context, uint32_t& dtypeKey, uint32_t& dataPerBlock)
{
    auto data_type = context->GetInputDesc(0)->GetDataType();
    dtypeKey = DTYPE_KEY_FP32;
    SetByDtype(data_type, dtypeKey, dataPerBlock);
    return data_type;
}

static void SetTilingParameters(
    RmsNormTilingData* tiling, uint32_t num_row, uint32_t num_col, uint32_t numColAlign,
    uint32_t block_factor, uint32_t lastBlockFactor, uint32_t row_factor,
    uint32_t ub_factor, float epsilon)
{
    const float avgFactor = (num_col == 0) ? 0 : 1.0f / num_col;
    tiling->set_num_row(num_row);
    tiling->set_num_col(num_col);
    tiling->set_num_col_align(numColAlign);
    tiling->set_block_factor(block_factor);
    tiling->set_last_block_factor(lastBlockFactor);
    tiling->set_row_factor(row_factor);
    tiling->set_ub_factor(ub_factor);
    tiling->set_epsilon(epsilon);
    tiling->set_avg_factor(avgFactor);
}

static void SaveTilingData(
    gert::TilingContext* context, RmsNormTilingData* tiling, uint32_t dtypeKey)
{
    const uint32_t tilingKey = dtypeKey;
    context->SetTilingKey(tilingKey);
    tiling->SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
}

static void SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* workspaceSizes = context->GetWorkspaceSizes(0);
    *workspaceSizes = 0;
}

static void LogTilingResults(
    gert::TilingContext* context, RmsNormTilingData* tiling, uint32_t dtypeKey,
    uint32_t useCoreNum, float epsilon)
{
    OPS_LOG_I(context, "Tiling Key: %u", dtypeKey);
    OPS_LOG_I(context, "Block Dim: %u", useCoreNum);
    OPS_LOG_I(context,
        "num_row: %u, num_col: %u, block_factor: %u, row_factor: %u, ub_factor: %u, epsilon: %f, avg_factor: %f",
        tiling->get_num_row(), tiling->get_num_col(), tiling->get_block_factor(),
        tiling->get_row_factor(), tiling->get_ub_factor(), epsilon, tiling->get_avg_factor());
}

static ge::graphStatus Tiling4RmsNorm(gert::TilingContext* context)
{
    OPS_LOG_D(context, "Tiling4RmsNorm running.");
    OP_CHECK_IF(!CheckInputOutputShape(context), OP_LOGE(context, "Input shape invalid"), return ge::GRAPH_FAILED);

    RmsNormTilingData tiling;

    uint32_t num_core;
    uint64_t ub_size;
    GetCompileParameters(context, num_core, ub_size);
    uint32_t num_row;
    uint32_t num_col;
    CalculateRowAndColParameters(context, num_row, num_col);
    float epsilon = 0;
    ge::graphStatus ret = GetEpsilonParameter(context, epsilon);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    uint32_t block_factor;
    uint32_t lastBlockFactor;
    uint32_t use_core_num;
    CalculateBlockParameters(num_row, num_core, block_factor, lastBlockFactor, use_core_num);
    context->SetBlockDim(use_core_num);

    uint32_t dtype_key;
    uint32_t data_per_block;
    ge::DataType data_type = SetDataTypeParameters(context, dtype_key, data_per_block);

    uint32_t row_factor = 64;
    uint32_t ub_factor = (dtype_key == DTYPE_KEY_FP32) ? UB_FACTOR_FP32 : UB_FACTOR_FP16;

    uint32_t numColAlign = ((num_col + data_per_block - 1) / data_per_block) * data_per_block;

    uint32_t rowLoop = (block_factor + row_factor - 1) / row_factor;
    uint32_t lastBlockRowLoop = (lastBlockFactor + row_factor - 1) / row_factor;
    uint32_t rowTail = block_factor - (rowLoop - 1) * row_factor;
    uint32_t lastBlockRowTail = lastBlockFactor - (lastBlockRowLoop - 1) * row_factor;

    tiling.set_row_loop(rowLoop);
    tiling.set_last_block_row_loop(lastBlockRowLoop);
    tiling.set_row_tail(rowTail);
    tiling.set_last_block_row_tail(lastBlockRowTail);

    SetTilingParameters(&tiling, num_row, num_col, numColAlign, block_factor,
        lastBlockFactor, row_factor, ub_factor, epsilon);
    SaveTilingData(context, &tiling, dtype_key);
    SetWorkspaceSize(context);
    LogTilingResults(context, &tiling, dtype_key, use_core_num, epsilon);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4RmsNorm(gert::TilingParseContext* context)
{
    OPS_LOG_D(context, "TilingPrepare4RmsNorm running.");
    auto compileInfo = context->GetCompiledInfo<RmsNormCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNorm).Tiling(Tiling4RmsNorm).TilingParse<RmsNormCompileInfo>(TilingPrepare4RmsNorm);
}