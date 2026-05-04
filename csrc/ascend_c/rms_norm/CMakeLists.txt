cmake_minimum_required(VERSION 3.14)
project(rms_norm_ascend_c CXX C ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -fPIC")
set(CMAKE_CUDA_FLAGS "-O3")

if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "ASCEND_TOOLKIT_HOME: ${ASCEND_HOME}")
else()
    message(FATAL_ERROR "ASCEND_TOOLKIT_HOME environment variable is not set")
endif()

set(CMAKE_SKIP_RPATH TRUE)

set(SYSROOT_PATH ${ASCEND_HOME}/ccec_compiler)

set(CMAKE_C_COMPILER ${ASCEND_HOME}/ccec_compiler/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${ASCEND_HOME}/ccec_compiler/bin/aarch64-linux-gnu-g++)
set(CMAKE_ASM_COMPILER ${ASCEND_HOME}/ccec_compiler/bin/aarch64-linux-gnu-gas)


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

set(OP_KERNEL_DIRS op_kernel)

file(GLOB_RECURSE OP_KERNEL_SOURCES ${OP_KERNEL_DIRS}/*.cpp)

file(GLOB OP_HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp)

set(OP_SOURCE ${OP_KERNEL_SOURCES} ${OP_HOST_SOURCES})

set(PYTHON_BINDING_SOURCE rms_norm_python_binding.cpp)

add_library(rms_norm_norm SHARED ${OP_SOURCE})

add_library(rms_norm_custom SHARED ${PYTHON_BINDING_SOURCE})

target_link_libraries(rms_norm_norm
    ascendcl
    register
    ascend_hal
    c_sec
    runtime
    ge_common
    ge_runner
    ge_executor
    log
    msprof
    msprofiler
    -fopenmp
)

target_link_libraries(rms_norm_custom
    torch_cpu
    torch_python
    c10
    -fopenmp
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)
install(TARGETS rms_norm_custom LIBRARY DESTINATION lib)
install(TARGETS rms_norm_norm LIBRARY DESTINATION lib)

set_target_properties(rms_norm_custom PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(rms_norm_norm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)