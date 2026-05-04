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
 * \file rms_norm_infershape.cpp
 * \brief RMS Norm operator infershape
 */
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_shape_registry.h"

namespace ops {
namespace {
using namespace ge;

Status InferShapeShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* gamma_shape = context->GetInputShape(1);

    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    size_t x_dim_num = x_shape->GetDimNum();
    size_t gamma_dim_num = gamma_shape->GetDimNum();

    OP_CHECK_IF(x_dim_num < 1, OP_LOGE(context, "Input x dimension must be at least 1"), return GRAPH_PARAM_INVALID);
    OP_CHECK_IF(gamma_dim_num != 1, OP_LOGE(context, "Input gamma dimension must be 1"), return GRAPH_PARAM_INVALID);

    int64_t hidden_size = gamma_shape->GetDim(0);

    OP_CHECK_IF(hidden_size <= 0, OP_LOGE(context, "Input gamma size must be positive"), return GRAPH_PARAM_INVALID);
    OP_CHECK_IF(x_dim_num < 1, OP_LOGE(context, "Input x dimension must be at least 1"), return GRAPH_PARAM_INVALID);
    OP_CHECK_IF(x_shape->GetDim(x_dim_num - 1) != hidden_size,
        OP_LOGE(context, "Input x last dimension (%ld) must match gamma size (%ld)",
                 x_shape->GetDim(x_dim_num - 1), hidden_size),
        return GRAPH_PARAM_INVALID);

    context->GetOutputShape(0)->SetShape(*x_shape);

    return GRAPH_SUCCESS;
}
}

IMPL_OP(RmsNorm)
    .InferShapeFunction(InferShapeShape);
}