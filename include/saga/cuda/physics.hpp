#pragma once
#include "saga/cuda/core.hpp"

namespace saga::cuda {

  /* !\brief Determine whether an index has occurrences in *in* before itself
   */
  template <class View>
  __global__ void sanitize_collision_index(View out, View in) {

    extern __shared__ char shared_memory[];

    auto shared_objects = (typename View::value_type *)shared_memory;

    auto bid = blockIdx.x * blockDim.x;
    auto gtid = bid + threadIdx.x;

    // we just need to process the first tiles
    auto size = in.size() > bid ? bid + blockDim.x : in.size();
    auto ntiles = size / blockDim.x + (size % blockDim.x != 0);

    auto current = gtid < in.size() ? in[gtid] : SAGA_CUDA_INVALID_INDEX;

    bool is_valid = true;

    // loop over the tiles
    for (auto tile = 0u; tile < ntiles; ++tile) {

      auto idx = tile * blockDim.x + threadIdx.x;

      if (idx < in.size())
        shared_objects[threadIdx.x] = in[idx];

      // the shared memory has been filled
      __syncthreads();

      if (gtid < in.size() && is_valid && tile * blockDim.x < gtid) {
        for (auto i = 0u; i < blockDim.x; ++i) {

          if (tile * blockDim.x + i < gtid) {

            if (shared_objects[i] == gtid || shared_objects[i] == current) {
              is_valid = false;
              break;
            }

          } else
            break;
        }
      }

      __syncthreads();
    }

    if (gtid < in.size())
      out[gtid] = is_valid ? current : SAGA_CUDA_INVALID_INDEX;
  }

  /* !\brief Determine the position of a set of indices after taking into
     account invalid entries

     Both vectors have the same length. Given an empty vector *out* and a vector
     *in*, the array is filled according to:

     in = {0, -1, 20, -1, 14, 31}
     out = {0, -1, 1, -1, 2, 3}
   */
  template <class View>
  __global__ void sanitize_merging_collision_index(View out, View in) {

    extern __shared__ char shared_memory[];

    auto shared_objects = (typename View::value_type *)shared_memory;

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // the number of tiles is equal to the dimension of the block
    auto ntiles = in.size() / blockDim.x + (in.size() % blockDim.x != 0);

    typename View::value_type result =
        (gtid < in.size() && in[gtid] != SAGA_CUDA_INVALID_INDEX)
            ? in[gtid]
            : SAGA_CUDA_INVALID_INDEX;

    // loop over the tiles
    for (auto tile = 0u; tile < ntiles; ++tile) {

      auto idx = tile * blockDim.x + threadIdx.x;

      if (idx < in.size())
        shared_objects[threadIdx.x] = in[idx];

      // the shared memory has been filled
      __syncthreads();

      if (result != SAGA_CUDA_INVALID_INDEX && gtid < in.size() &&
          tile * blockDim.x < gtid) {
        for (auto i = 0u; i < blockDim.x; ++i) {

          if (tile * blockDim.x + i < gtid) {

            if (shared_objects[i] != SAGA_CUDA_INVALID_INDEX)
              ++result;

          } else
            break;
        }
      }

      __syncthreads();
    }

    if (gtid < in.size())
      out[gtid] = result;
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

  template <class ParticlesView, class IndicesView, class Functor,
            class... Args>
  __global__ void evaluate_collisions(ParticlesView particles,
                                      IndicesView indices, Functor functor,
                                      Args... args) {

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid < particles.size() && indices[gtid] != SAGA_CUDA_INVALID_INDEX)
      functor(particles[gtid], particles[indices[gtid]], args...);
  }
} // namespace saga::cuda
