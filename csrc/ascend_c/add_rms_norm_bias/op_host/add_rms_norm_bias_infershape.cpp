/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * AddRmsNormBias operator shape inference.
 */

#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_shape_registry.h"

namespace ops {
namespace {
using namespace ge;

Status InferShapeShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* gamma_shape = context->GetInputShape(2);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    size_t x1_dim_num = x1_shape->GetDimNum();
    size_t gamma_dim_num = gamma_shape->GetDimNum();

    OP_CHECK_IF(x1_dim_num < 1,
        OP_LOGE(context, "Input x1 dimension must be at least 1"), return GRAPH_PARAM_INVALID);
    OP_CHECK_IF(gamma_dim_num != 1,
        OP_LOGE(context, "Input gamma dimension must be 1"), return GRAPH_PARAM_INVALID);

    int64_t hidden_size = gamma_shape->GetDim(0);
    OP_CHECK_IF(hidden_size <= 0,
        OP_LOGE(context, "Input gamma size must be positive"), return GRAPH_PARAM_INVALID);
    OP_CHECK_IF(x1_shape->GetDim(x1_dim_num - 1) != hidden_size,
        OP_LOGE(context, "Input x1 last dimension (%ld) must match gamma size (%ld)",
                 x1_shape->GetDim(x1_dim_num - 1), hidden_size),
        return GRAPH_PARAM_INVALID);

    // y has same shape as x1
    context->GetOutputShape(0)->SetShape(*x1_shape);

    // rstd shape: same as x1 but last dims replaced with 1
    gert::Shape rstd_shape;
    for (size_t i = 0; i < x1_dim_num - gamma_dim_num; ++i) {
        rstd_shape.AppendDim(x1_shape->GetDim(i));
    }
    for (size_t i = 0; i < gamma_dim_num; ++i) {
        rstd_shape.AppendDim(1);
    }
    context->GetOutputShape(1)->SetShape(rstd_shape);

    // x (residual output) has same shape as x1
    context->GetOutputShape(2)->SetShape(*x1_shape);

    return GRAPH_SUCCESS;
}
}

IMPL_OP(AddRmsNormBias)
    .InferShapeFunction(InferShapeShape);
}
