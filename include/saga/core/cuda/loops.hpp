#pragma once
#include "saga/core/cuda/core.hpp"

namespace saga::core::cuda {

  namespace detail {
    template <class Particles, class Function, class... Args>
    __global__ void apply_simple_function_inplace(Particles &particles,
                                                  Function &&function,
                                                  Args &&...args) {
      auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gtid < particles.size())
        function(particles[gtid], args...);
    }
  } // namespace detail

  template <class Particles, class Function, class... Args>
  __global__ void
  apply_contiguous_function_inplace(Particles &particles, Forces const &forces,
                                    Function &&function, Args &&...args) {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < particles.size())
      function(particles[gtid], forces[gtid], args...);
  }

  template <class Particles, class Function, class... Args>
  void apply_simple_function_inplace(Particles &particles, Function &&function,
                                     Args &&...args) {

    auto N = particles.size();
    auto nblocks = N / SAGA_CUDA_MAX_THREADS_PER_BLOCK_X +
                   (N % SAGA_CUDA_MAX_THREADS_PER_BLOCK_X != 0);

    detail::apply_simple_function_inplace<<<
        nblocks, SAGA_CUDA_MAX_THREADS_PER_BLOCK_X>>>(particles, args...);
  }

  /// Kernel function to evaluate a function that calculates the force
  template <class Forces, class Function, class Particles>
  __global__ void add_forces(std::size_t tile_size, Forces &forces,
                             Function &&force_function,
                             Particles const &particles) {

    extern __shared__ typename Particles::value_type[] shared_particles;

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    auto current_particle = gtid < particles.size()
                                ? particles[gtid]
                                : particles[0]; // default to the first particle

    typename Forces::value_type acc = {0.f, 0.f, 0.f, 0.f};

    for (auto i = 0u, tile = 0u; i < particles.size(); i += tile_size, ++tile) {

      auto idx = tile * blockDim.x + threadIdx.x;

      shared_particles[threadIdx.x] = particles[idx];

      __syncthreads();

      if (idx < particles.size()) {
        for (auto i = 0u; i < blockDim.x; ++i) {
          if (tile * blockDim.x + i == gtid)
            continue;
          else if (tile * blockDim.x + i < particles.size())
            acc += force_function(particle_position, shared_particles[i]);
          else
            break;
        }
      }

      __syncthreads();
    }

    if (gtid < particles.size())
      forces[gtid] += acc;
  }
} // namespace saga::core::cuda
