#pragma once
#include "saga/core/cuda/core.hpp"

#include <stdexcept>
#include <string>

namespace saga::core::cuda {

  namespace detail {
    template <class ParticlesView, class Functor, class... Args>
    __global__ void apply_simple_functor_inplace(ParticlesView particles,
                                                 Functor const &functor,
                                                 Args... args) {
      auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gtid < particles.size())
        functor(particles[gtid], args...);
    }
  } // namespace detail

  template <class ParticlesView, class ConstForcesView, class Functor,
            class... Args>
  __global__ void apply_contiguous_functor_inplace(ParticlesView particles,
                                                   ConstForcesView forces,
                                                   Functor const &functor,
                                                   Args... args) {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < particles.size())
      functor(particles[gtid], forces[gtid], args...);
  }

  template <class ParticlesView, class Functor, class... Args>
  void apply_simple_functor_inplace(ParticlesView particles,
                                    Functor const &functor, Args &&...args) {

    auto N = particles.size();
    auto nblocks = N / SAGA_CUDA_MAX_THREADS_PER_BLOCK_X +
                   (N % SAGA_CUDA_MAX_THREADS_PER_BLOCK_X != 0);

    detail::apply_simple_functor_inplace<<<nblocks,
                                           SAGA_CUDA_MAX_THREADS_PER_BLOCK_X>>>(
        particles, functor, args...);

    auto code = cudaPeekAtLastError();
    if (code != cudaSuccess)
      throw std::runtime_error(
          "Failed to evaluate functor on " + std::to_string(N) +
          " objects, with " + std::to_string(nblocks) + " block(s) and " +
          std::to_string(SAGA_CUDA_MAX_THREADS_PER_BLOCK_X) +
          " threads per block. Reason: " +
          std::string{cudaGetErrorString(code)});
  }

  /// Kernel functor to evaluate a functor that calculates the force
  template <class ForcesView, class Functor, class ConstParticlesView>
  __global__ void add_forces(std::size_t tile_size, ForcesView &forces,
                             Functor const &force_functor,
                             ConstParticlesView particles) {

    extern __shared__ char shared_memory[];

    // CUDA does not allow to create arrays of template types yet
    auto shared_particles =
        (typename ConstParticlesView::value_type *)shared_memory;

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    auto current_particle = gtid < particles.size()
                                ? particles[gtid]
                                : particles[0]; // default to the first particle

    typename ForcesView::value_type acc = {0.f, 0.f, 0.f};

    for (auto i = 0u, tile = 0u; i < particles.size(); i += tile_size, ++tile) {

      auto idx = tile * blockDim.x + threadIdx.x;

      shared_particles[threadIdx.x] = particles[idx];

      __syncthreads();

      if (idx < particles.size()) {
        for (auto i = 0u; i < blockDim.x; ++i) {
          if (tile * blockDim.x + i == gtid)
            continue;
          else if (tile * blockDim.x + i < particles.size()) {
            auto r = force_functor(current_particle, shared_particles[i]);
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
