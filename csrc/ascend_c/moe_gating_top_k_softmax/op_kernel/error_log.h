#ifndef OPS_BUILT_IN_OP_KERNEL_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_KERNEL_ERROR_LOG_H_

#include <string>
#include "toolchain/slog.h"

#define KERNEL_LOGE(...) printf("[ERROR][%s] ", __VA_ARGS__); printf("\n")

#endif  // OPS_BUILT_IN_OP_KERNEL_ERROR_LOG_H_