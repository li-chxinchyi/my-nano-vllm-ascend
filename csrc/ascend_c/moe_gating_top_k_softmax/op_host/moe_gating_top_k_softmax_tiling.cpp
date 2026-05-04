/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
* \file moe_gating_top_k_softmax_tiling.cpp
* \brief Tiling implementation for NpuMoeGatingTopKSoftmax
*/
#include <cmath>
#include "register/op_def_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"

#include "moe_gating_top_k_softmax_tiling.h"

#ifndef CEIL_ALIGN
#define CEIL_ALIGN(val, align) ((((val) + (align) - 1) / (align)) * (align))
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

namespace optiling {
const static int64_t X_INPUT_DIMS = 2;
const static int64_t BIAS_INPUT_DIMS = 1;
const static int64_t Y_OUTPUT_DIMS = 2;
const static int64_t EXPERT_IDX_OUTPUT_DIMS = 2;
const static int64_t OUT_OUTPUT_DIMS = 2;
const static int64_t MAX_EXPERT_COUNT = 2048;

const static int64_t X_INPUT_INDEX = 0;
const static int64_t BIAS_INPUT_INDEX = 1;
const static int64_t Y_OUTPUT_INDEX = 0;
const static int64_t EXPERT_IDX_OUTPUT_INDEX = 1;
const static int64_t OUT_OUTPUT_INDEX = 2;
const static int64_t K_ATTR_INDEX = 0;
const static int64_t SCALING_FACTOR_ATTR_INDEX = 1;
const static int64_t EPS_ATTR_INDEX = 2;
const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216;

constexpr int32_t ROW_COUNT_PER_TASK = 1;

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(4)));
}

class MoeGatingTopKSoftmaxTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeGatingTopKSoftmaxTilingBase(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeGatingTopKSoftmaxTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;

    ge::graphStatus GetShapeAttrsInfo() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus DoLibApiTiling() override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    void SplitRows();
    void CalTmpBufUbSize();

    const gert::Shape *xShape_ = nullptr;
    const gert::Shape *biasShape_ = nullptr;
    const gert::Shape *yShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outShape_ = nullptr;

    int64_t rows_ = 0;
    int64_t expertCount_ = 0;
    int64_t addBias_ = 0;

    int64_t k_ = 0;
    float routedScalingFactor_ = 1.0;
    float eps_ = 1e-20f;

    int64_t inputDtypeSize_;
    const char *opName_ = "";
    MoeGatingTopKSoftmaxTilingData moeGatingTopKSoftmaxTilingData_;
};

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::CheckInputShape()
{
    size_t xDimNum = xShape_->GetDimNum();

    OP_CHECK_IF(xDimNum != X_INPUT_DIMS,

    OP_LOGE(context_, "The dim number of x is: %zu, but should be %zu.", xDimNum, X_INPUT_DIMS),
            return ge::GRAPH_FAILED);

    rows_ = xShape_->GetDim(0);
    expertCount_ = xShape_->GetDim(1);

    moeGatingTopKSoftmaxTilingData_.set_rowCount(rows_);
    moeGatingTopKSoftmaxTilingData_.set_expertCount(expertCount_);
    if (biasShape_ != nullptr) {
        addBias_ = 1;
        size_t biasDimNum = biasShape_->GetDimNum();
        OP_CHECK_IF(biasDimNum != BIAS_INPUT_DIMS,
                    OP_LOGE(context_, "The dim number of bias is: %zu, but should be %zu.", biasDimNum, BIAS_INPUT_DIMS),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            biasShape_->GetDim(0) != expertCount_,
            OP_LOGE(context_, "The first dim of bias is: %ld, but should be %ld.", biasShape_->GetDim(0), expertCount_),
            return ge::GRAPH_FAILED);
    }
    moeGatingTopKSoftmaxTilingData_.set_addBias(addBias_);

    OP_CHECK_IF(k_ > expertCount_,
                OP_LOGE(context_, "k is: %ld, expert num is: %ld, k cannot be greater than expert num.", k_, expertCount_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::CheckAttr()
{
    OP_CHECK_IF(
        expertCount_ > MAX_EXPERT_COUNT,
        OP_LOGE(context_, "expert count is: %ld, but should not greater than %ld.", expertCount_, MAX_EXPERT_COUNT),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(k_ <= 0, OP_LOGE(context_, "k is: %ld, but should be greater than 0.", k_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    OP_LOGI(context_, "GetShapeAttrsInfo: opName = %s", opName_);
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    xShape_ = &xShapePtr->GetStorageShape();
    OP_LOGI(context_, "xShape: %s", xShape_->ToString().c_str());

    auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INPUT_INDEX);
    biasShape_ = biasShapePtr == nullptr ? nullptr : &biasShapePtr->GetStorageShape();
    if (biasShape_ != nullptr) {
        OP_LOGI(context_, "biasShape: %s", biasShape_->ToString().c_str());
    }

    auto yShapePtr = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    yShape_ = &yShapePtr->GetStorageShape();
    OP_LOGI(context_, "yShape: %s", yShape_->ToString().c_str());
    auto expertIdxPtr = context_->GetOutputShape(EXPERT_IDX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxPtr);
    expertIdxShape_ = &expertIdxPtr->GetStorageShape();
    OP_LOGI(context_, "expertIdxShape: %s", expertIdxShape_->ToString().c_str());
    auto outPtr = context_->GetOutputShape(OUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outPtr);
    outShape_ = &outPtr->GetStorageShape();
    if (outShape_ != nullptr) {
        OP_LOGI(context_, "outShape: %s", outShape_->ToString().c_str());
    }

    auto x = context_->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    auto xDtype = x->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16),
        OP_LOGE(context_, "x dtype %s error, only supports float32, half, bf16. please check.",
             ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    if (biasShapePtr != nullptr) {
        auto biasDtype = context_->GetOptionalInputDesc(BIAS_INPUT_INDEX)->GetDataType();
        OP_LOGI(context_, "bias dtype: %s", ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str());
        OP_CHECK_IF((biasDtype != xDtype),
                    OP_LOGE(context_, "bias dtype %s not equal x dtype %s, please check.",
                         ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                         ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    auto yDesc = context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_LOGI(context_, "y dtype: %s", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str());
    OP_CHECK_IF((yDtype != xDtype),
                OP_LOGE(context_, "y out dtype %s must be the same with x dtype %s.",
                     ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                     ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto expertIdDesc = context_->GetOutputDesc(EXPERT_IDX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdDesc);
    auto expertIdDtype = expertIdDesc->GetDataType();
    OP_LOGI(context_, "expertId dtype: %s", ge::TypeUtils::DataTypeToSerialString(expertIdDtype).c_str());
    OP_CHECK_IF((expertIdDtype != ge::DataType::DT_INT32),
                OP_LOGE(context_, "expertId out dtype %s error, only supports int32. please check.",
                     ge::TypeUtils::DataTypeToSerialString(expertIdDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto normOutDesc = context_->GetOutputDesc(OUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, normOutDesc);
    auto normOutDtype = normOutDesc->GetDataType();
    OP_CHECK_IF((normOutDtype != ge::DataType::DT_FLOAT),
                OP_LOGE(context_, "norm out dtype %s error, only supports float. please check.",
                     ge::TypeUtils::DataTypeToSerialString(normOutDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *kPtr = attrs->GetAttrPointer<int64_t>(K_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kPtr);
    k_ = *kPtr;
    OP_LOGI(context_, "Attr k is: %ld", k_);
    moeGatingTopKSoftmaxTilingData_.set_k(k_);

    const float *routedScalingFactorPtr = attrs->GetAttrPointer<float>(SCALING_FACTOR_ATTR_INDEX);
    if (routedScalingFactorPtr != nullptr) {
        routedScalingFactor_ = *routedScalingFactorPtr;
        OP_LOGI(context_, "Attr routed_scaling_factor is: %f", routedScalingFactor_);
        moeGatingTopKSoftmaxTilingData_.set_routedScalingFactor(routedScalingFactor_);
    }
    OP_LOGI(context_, "Attr routed_scaling_factor is: %f ", routedScalingFactor_);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
        OP_LOGI(context_, "Attr eps is: %f", eps_);
        moeGatingTopKSoftmaxTilingData_.set_eps(eps_);
    }
    OP_LOGI(context_, "Attr eps is: %f ", eps_);

    inputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    OP_LOGI(context_, "inputDtypeSize_: %ld", inputDtypeSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    OP_LOGI(context_, "GetPlatformInfo: blockDim = %ld, ubSize = %lu", aicoreParams_.blockDim, aicoreParams_.ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::CheckOutShape()
{
    OP_LOGI(context_, "CheckOutShape: yShape_: %s, xShape_: %s", yShape_->ToString().c_str(), xShape_->ToString().c_str());
    OP_CHECK_IF((yShape_->GetDimNum() != xShape_->GetDimNum()),
                OP_LOGE(context_, "y out shape num %zu and x shape num %zu not equal, please check.", yShape_->GetDimNum(),
                     xShape_->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((expertIdxShape_->GetDimNum() != xShape_->GetDimNum()),
                OP_LOGE(context_, "expertId out shape num %zu and x shape num %zu not equal, please check.",
                     expertIdxShape_->GetDimNum(), xShape_->GetDimNum()),
                return ge::GRAPH_FAILED);
    if (outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDimNum() != xShape_->GetDimNum()),
                    OP_LOGE(context_, "norm out shape num %zu and x shape num %zu not equal, please check.",
                         outShape_->GetDimNum(), xShape_->GetDimNum()),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((yShape_->GetDim(0) != xShape_->GetDim(0)),
                OP_LOGE(context_, "y out dim[0] %ld not equal x dim[0] %ld, please check.", yShape_->GetDim(0),
                     xShape_->GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((expertIdxShape_->GetDim(0) != xShape_->GetDim(0)),
                OP_LOGE(context_, "expertId out dim[0] %ld not equal x dim[0] %ld, please check.",
                     expertIdxShape_->GetDim(0), xShape_->GetDim(0)),
                return ge::GRAPH_FAILED);
    if (outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDim(0) != xShape_->GetDim(0)),
                    OP_LOGE(context_, "norm out dim[0] %ld and x dim[0] %ld not equal, please check.",
                         outShape_->GetDim(0), outShape_->GetDim(0)),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((yShape_->GetDim(1) != k_),
                OP_LOGE(context_, "y dim[1] %ld not equal k %ld, please check.", yShape_->GetDim(1), k_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((expertIdxShape_->GetDim(1) != k_),
                OP_LOGE(context_, "expertId dim[1] %ld not equal k %ld, please check.", expertIdxShape_->GetDim(1), k_),
                return ge::GRAPH_FAILED);
    if (outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDim(1) != xShape_->GetDim(1)),
                    OP_LOGE(context_, "normOut dim[1] %ld and x dim[1] %ld not equal, please check.", outShape_->GetDim(1),
                         xShape_->GetDim(1)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKSoftmaxTilingBase::SplitRows()
{
    int64_t perCoreRows = CEIL_DIV(rows_, static_cast<int64_t>(aicoreParams_.blockDim));
    int64_t needCoreNum = CEIL_DIV(rows_, perCoreRows);
    int64_t lastCoreRows = rows_ % perCoreRows == 0 ? perCoreRows : rows_ % perCoreRows;
    moeGatingTopKSoftmaxTilingData_.set_needCoreNum(needCoreNum);
    moeGatingTopKSoftmaxTilingData_.set_perCoreRowCount(perCoreRows);
    moeGatingTopKSoftmaxTilingData_.set_lastCoreRowCount(lastCoreRows);
}

void MoeGatingTopKSoftmaxTilingBase::CalTmpBufUbSize()
{
    int64_t indexTmpBuf = (expertCount_ + 31) / 32 * 32 * static_cast<int64_t>(sizeof(float));
    moeGatingTopKSoftmaxTilingData_.set_calTmpBufUbSize(std::max(indexTmpBuf, static_cast<int64_t>(4096)));
}

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::DoOpTiling()
{
    OP_LOGI(context_, "DoOpTiling: start");
    auto ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    CalTmpBufUbSize();
    SplitRows();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxTilingBase::PostTiling()
{
    context_->SetBlockDim(moeGatingTopKSoftmaxTilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    moeGatingTopKSoftmaxTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeGatingTopKSoftmaxTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKSoftmaxTilingBase::Reset()
{
    opName_ = nullptr;
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKSoftmax, MoeGatingTopKSoftmaxTilingBase, 2000);
} // namespace optiling