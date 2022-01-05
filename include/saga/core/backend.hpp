#pragma once
#include <stdexcept>

namespace saga {

  enum class backend { CPU, CUDA };

#if SAGA_CUDA_ENABLED
#define __saga_core_function__ __device__ __host__
#define SAGA_CHECK_LAS_ERROR(message)                                          \
  {                                                                            \
    auto code = cudaPeekAtLastError();                                         \
    if (code != cudaSuccess)                                                   \
      throw std::runtime_error(std::string{message} + ". Reason: " +           \
                               std::string{cudaGetErrorString(code)});         \
  }
#else
#define __saga_core_function__
#define SAGA_THROW_CUDA_ERROR                                                  \
  throw std::runtime_error("Attempt to run with CUDA backend on software "     \
                           "compiled exclusively for CPU")
#endif
} // namespace saga
