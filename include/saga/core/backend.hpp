#pragma once
#if SAGA_CUDA_ENABLED
#include "saga/core/cuda/vector.hpp"
#endif

namespace saga {

  enum class backend { CPU, CUDA };

#if SAGA_CUDA_ENABLED
#define __saga_core_function__ __device__ __host__
#else
#define __saga_core_function__
#endif
} // namespace saga
