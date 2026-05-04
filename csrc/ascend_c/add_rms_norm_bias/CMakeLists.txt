cmake_minimum_required(VERSION 3.14)
project(add_rms_norm_bias_ascend_c CXX C ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -fPIC")

if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "ASCEND_TOOLKIT_HOME: ${ASCEND_HOME}")
else()
    message(FATAL_ERROR "ASCEND_TOOLKIT_HOME environment variable is not set")
endif()

set(CMAKE_SKIP_RPATH TRUE)
set(ASCEND_INSTALL_DIR ${ASCEND_HOME})

include_directories(
    ${ASCEND_INSTALL_DIR}/include
    ${ASCEND_INSTALL_DIR}/include/levels
    ${ASCEND_INSTALL_DIR}/include/graph
    ${ASCEND_INSTALL_DIR}/include/operator
    ${ASCEND_INSTALL_DIR}/include/ge_plugin/custom_opapi
    ${ASCEND_INSTALL_DIR}/compiler/opskernel/built-in/op_proto/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

link_directories(
    ${ASCEND_INSTALL_DIR}/lib64
)

file(GLOB_RECURSE OP_KERNEL_SOURCES op_kernel/*.cpp)
file(GLOB OP_HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp)

set(OP_SOURCE ${OP_KERNEL_SOURCES} ${OP_HOST_SOURCES})

add_library(add_rms_norm_bias_norm SHARED ${OP_SOURCE})
add_library(add_rms_norm_bias_custom SHARED add_rms_norm_bias_true_npu.cpp)

target_link_libraries(add_rms_norm_bias_norm
    ascendcl register ascend_hal c_sec runtime
    ge_common ge_runner ge_executor log msprof msprofiler
    -fopenmp
)

target_link_libraries(add_rms_norm_bias_custom
    torch_cpu torch_python c10
    -fopenmp
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)
install(TARGETS add_rms_norm_bias_custom LIBRARY DESTINATION lib)
install(TARGETS add_rms_norm_bias_norm LIBRARY DESTINATION lib)
