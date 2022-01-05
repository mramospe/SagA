#pragma once
#include "saga/core/cuda/core.hpp"

#include <stdexcept>
#include <string>

namespace saga::core::cuda {

  /// Apply a functor on the given views
  template <class View, class Functor, class... Args>
  __global__ void apply_functor(View obj, Functor functor, Args... args) {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < obj.size())
      functor(obj[gtid], args...);
  }

  /// Apply a functor using the information from the two views
  template <class FirstView, class SecondView, class Functor, class... Args>
  __global__ void
  apply_functor_contiguous_views(FirstView first, SecondView second,
                                 Functor functor, Args... args) {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < first.size())
      functor(first[gtid], second[gtid], args...);
  }

  /// Set the values of a vector
  struct set_vector_value {
    template <class T>
    __saga_core_function__ void operator()(T &v, T def) const {
      v = def;
    }
  };

  /// Set forces to zero
  template <class View, class ValueType>
  __global__ void set_view_values(View obj, ValueType def) {

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid < obj.size())
      obj[gtid] = def;
  }

  /* !\brief Kernel functor to evaluate a functor that calculates the force

     Here, each block calculates the interaction on a tile of d * d where d is
     the dimension of the block. In each step, each thread updates the
     corresponding slot in the shared memory with the information from one of
     the particles and, after synchronizing the threads, iterates over the
     shared memory to calculate the interaction with each particle. Once this is
     done, the threads are synchronized and the process is repeated as many
     times as tiles are created.
   */
  template <class ForcesView, class Functor, class ConstParticlesView>
  __global__ void calculate_forces(ForcesView forces, Functor force_functor,
                                   ConstParticlesView particles) {

    extern __shared__ char shared_memory[];

    // CUDA does not allow to create arrays of template types yet
    auto shared_particles =
        (typename ConstParticlesView::value_type *)shared_memory;

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    auto current_particle = gtid < particles.size()
                                ? particles[gtid]
                                : particles[0]; // default to the first particle

    // initial value
    typename ForcesView::value_type acc = {0.f, 0.f, 0.f};

    // the number of tiles is equal to the dimension of the block
    auto ntiles =
        particles.size() / blockDim.x + (particles.size() % blockDim.x != 0);

    // loop over the tiles
    for (auto tile = 0u; tile < ntiles; ++tile) {

      auto idx = tile * blockDim.x + threadIdx.x;

      if (idx < particles.size())
        shared_particles[threadIdx.x] = particles[idx];

      // the shared memory has been filled
      __syncthreads();

      if (gtid < particles.size()) {
        for (auto i = 0u; i < blockDim.x; ++i) {
          if (tile * blockDim.x + i == gtid)
            continue; // the particle is the one that is being processed
          else if (tile * blockDim.x + i < particles.size()) {
            // the particle is valid, its force is added
            auto r =
                force_functor(current_particle, shared_particles[threadIdx.x]);
            acc.set_x(acc.get_x() + r.get_x());
            acc.set_y(acc.get_y() + r.get_y());
            acc.set_z(acc.get_z() + r.get_z());
          } else
            // the next particles will also fall into this statement, so simply
            // stop looping over the array
            break;
        }
      }

      // the interactions have been computed
      __syncthreads();
    }

    if (gtid < particles.size()) {
      // fill the resulting force from the sum
      auto force_proxy = forces[gtid];
      force_proxy.set_x(force_proxy.get_x() + acc.get_x());
      force_proxy.set_y(force_proxy.get_y() + acc.get_y());
      force_proxy.set_z(force_proxy.get_z() + acc.get_z());
    }
  }
} // namespace saga::core::cuda
