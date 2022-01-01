#pragma once
#include "saga/core/cuda/core.hpp"

#include <stdexcept>
#include <string>

namespace saga::core::cuda {

  namespace detail {
    template <class Particles, class Functor, class... Args>
    __global__ void apply_simple_function_inplace(Particles &particles,
                                                  Functor const &function,
                                                  Args &&...args) {
      auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gtid < particles.size())
        function(particles[gtid], args...);
    }
  } // namespace detail

  template <class Particles, class Forces, class Functor, class... Args>
  __global__ void
  apply_contiguous_function_inplace(Particles &particles, Forces const &forces,
                                    Functor const &function, Args &&...args) {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < particles.size())
      function(particles[gtid], forces[gtid], args...);
  }

  template <class Particles, class Functor, class... Args>
  void apply_simple_function_inplace(Particles &particles,
                                     Functor const &function, Args &&...args) {

    auto N = particles.size();
    auto nblocks = N / SAGA_CUDA_MAX_THREADS_PER_BLOCK_X +
                   (N % SAGA_CUDA_MAX_THREADS_PER_BLOCK_X != 0);

    detail::apply_simple_function_inplace<<<
        nblocks, SAGA_CUDA_MAX_THREADS_PER_BLOCK_X>>>(particles, function,
                                                      args...);

    auto code = cudaPeekAtLastError();
    if (code != cudaSuccess)
      throw std::runtime_error(
          "Failed to evaluate function on " + std::to_string(N) +
          " objects, with " + std::to_string(nblocks) + " block(s) and " +
          std::to_string(SAGA_CUDA_MAX_THREADS_PER_BLOCK_X) +
          " threads per block. Reason: " +
          std::string{cudaGetErrorString(code)});
  }

  /// Kernel function to evaluate a function that calculates the force
  template <class Forces, class Functor, class Particles>
  __global__ void add_forces(std::size_t tile_size, Forces &forces,
                             Functor const &force_function,
                             Particles const &particles) {

    extern __shared__ char shared_memory[];

    // CUDA does not allow to create arrays of template types yet
    auto shared_particles = (typename Particles::value_type *)shared_memory;

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    auto current_particle = gtid < particles.size()
                                ? particles[gtid]
                                : particles[0]; // default to the first particle

    typename Forces::value_type acc = {0.f, 0.f, 0.f};

    for (auto i = 0u, tile = 0u; i < particles.size(); i += tile_size, ++tile) {

      auto idx = tile * blockDim.x + threadIdx.x;

      shared_particles[threadIdx.x] = particles[idx];

      __syncthreads();

      if (idx < particles.size()) {
        for (auto i = 0u; i < blockDim.x; ++i) {
          if (tile * blockDim.x + i == gtid)
            continue;
          else if (tile * blockDim.x + i < particles.size()) {
            auto r = force_function(current_particle, shared_particles[i]);
            acc.set_x(acc.get_x() + r.get_x());
            acc.set_y(acc.get_y() + r.get_y());
            acc.set_z(acc.get_z() + r.get_z());
          } else
            break;
        }
      }

      __syncthreads();
    }

    if (gtid < particles.size()) {
      auto force_proxy = forces[gtid];
      force_proxy.set_x(force_proxy.get_x() + acc.get_x());
      force_proxy.set_y(force_proxy.get_y() + acc.get_y());
      force_proxy.set_z(force_proxy.get_z() + acc.get_z());
    }
  }
} // namespace saga::core::cuda
