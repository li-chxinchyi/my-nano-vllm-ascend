import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, name_mapping=None, target_layer: str = None):
    """Load safetensors weights into the model (supports loading only specific layer).

    Args:
        model: Target torch module whose parameters will be filled.
        path: Directory containing *.safetensors files.
        name_mapping: Optional callable that maps a weight name to the
            corresponding parameter name. Returning ``None`` skips the weight.
        target_layer: Optional string specifying the layer name to load (supports fuzzy match).
                      If None, load all layers (original behavior).
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 遍历所有 safetensors 文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 核心修改：只处理包含目标层名称的权重
                if target_layer is not None and target_layer not in weight_name:
                    continue  # 跳过不匹配目标层的权重

                target_name = weight_name
                if name_mapping is not None:
                    target_name = name_mapping(target_name)
                    if target_name is None:
                        continue

                # 处理 packed modules 的权重映射
                for k, (v, shard_id) in packed_modules_mapping.items():
                    if k in weight_name:
                        param_name = target_name
                        if k in param_name:
                            param_name = param_name.replace(k, v)
                        elif k == "gate_proj" and "gate_up_proj" in param_name:
                            param_name = param_name.replace("gate_up_proj", v)
                        elif k == "up_proj" and "gate_up_proj" in param_name:
                            param_name = param_name.replace("gate_up_proj", v)

                        # 再次检查参数名是否包含目标层（防止映射后名称变化）
                        if target_layer is not None and target_layer not in param_name:
                            break

                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        tensor = f.get_tensor(weight_name)
                        # 对齐数据类型避免精度不匹配
                        if tensor.dtype != param.dtype:
                            tensor = tensor.to(param.dtype)
                        weight_loader(param, tensor, shard_id)
                        break
                else:
                    # 处理普通权重
                    try:
                        param = model.get_parameter(target_name)
                    except AttributeError as e:
                        raise AttributeError(
                            f"Failed to locate parameter '{target_name}' mapped from '{weight_name}'") from e

                    # 检查参数名是否包含目标层
                    if target_layer is not None and target_layer not in target_name:
                        continue

                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    tensor = f.get_tensor(weight_name)
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    weight_loader(param, tensor)
