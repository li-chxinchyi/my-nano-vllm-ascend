#!/bin/bash
# 编译 Ascend C RMS Norm kernel + torch 绑定
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_cmake"

export PATH="/usr/local/python3.11.14/bin:$PATH"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-8.5.0}"
export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_HOME}"

echo "========================================"
echo "  编译 Ascend C RMS Norm Kernel"
echo "========================================"
echo "  CANN: ${ASCEND_TOOLKIT_HOME}"
echo "  Python: $(which python3.11)"
echo "  Build: ${BUILD_DIR}"
echo ""

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

SOC_VERSION="${SOC_VERSION:-ascend910b}"
echo "  SOC: ${SOC_VERSION}"

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=/usr/local/python3.11.14/bin/python3.11 \
    -DSOC_VERSION=${SOC_VERSION} \
    -DRUN_MODE=npu \
    2>&1

make -j$(nproc) 2>&1

echo ""
echo "========================================"
echo "  编译完成"
echo "========================================"
ls -la "${SCRIPT_DIR}/build/"*.so 2>/dev/null || echo "  (检查 build/ 目录)"
