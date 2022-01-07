#pragma once
#include <tuple>

#define SAGA_CUDA_MAX_THREADS_PER_BLOCK_X 1024

namespace saga::cuda {

  /// Simple object to hold the number of blocks and threads
  struct grid_1d {
    std::size_t blocks;
    std::size_t threads_per_block;
  };

  /// Determine the optimal grid in one dimension
  template <class Container> grid_1d optimal_grid_1d(Container const &cont) {
    if (cont.size() < SAGA_CUDA_MAX_THREADS_PER_BLOCK_X)
      return {1, cont.size()};
    else {
      std::size_t blocks =
          cont.size() / SAGA_CUDA_MAX_THREADS_PER_BLOCK_X +
          (cont.size() % SAGA_CUDA_MAX_THREADS_PER_BLOCK_X != 0);
      return {blocks, SAGA_CUDA_MAX_THREADS_PER_BLOCK_X};
    }
  }
} // namespace saga::cuda
