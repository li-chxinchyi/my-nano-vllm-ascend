cmake_minimum_required(VERSION 3.16)
project(rms_norm_ascendc)

set(CMAKE_CXX_STANDARD 17)

# ============================================================
# 1. 找 Python / pybind11 / Torch
# ============================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

# torch_npu include
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

# ============================================================
# 2. CANN / Ascend C
# ============================================================
set(ASCEND_HOME_PATH $ENV{ASCEND_TOOLKIT_HOME})
if(NOT ASCEND_HOME_PATH)
    message(FATAL_ERROR "ASCEND_TOOLKIT_HOME not set")
endif()

# 找 ascendc.cmake
if(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found in CANN")
endif()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "ascend910b" CACHE STRING "SoC version")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ============================================================
# 3. 编译 Ascend C kernel (用 ccec 编译器)
# ============================================================
ascendc_library(rms_norm_kernel SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm_kernel.cpp
)

# ============================================================
# 4. 编译 torch 绑定 (用 g++ 编译器)
# ============================================================
add_library(rms_norm_ascendc SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm_ascendc_binding.cpp
)

target_include_directories(rms_norm_ascendc PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${ASCEND_HOME_PATH}/include
    ${Python3_INCLUDE_DIRS}
)

target_link_directories(rms_norm_ascendc PRIVATE ${TORCH_NPU_PATH}/lib)

target_link_libraries(rms_norm_ascendc PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    rms_norm_kernel
    Python3::Python
)

# 输出到 build/ 目录
set_target_properties(rms_norm_ascendc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    PREFIX ""
    SUFFIX ".so"
)
set_target_properties(rms_norm_kernel PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)
