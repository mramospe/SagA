#pragma once
#include "saga/core/backend.hpp"
#if SAGA_CUDA_ENABLED
#include "saga/core/cuda/core.hpp"
#include "saga/core/cuda/loops.hpp"
#endif

namespace saga::core {

  namespace detail {

    /// Core function to integrate the position of a particle
    template <class Proxy, class FloatType>
    __saga_core_function__ void integrate_position(Proxy &p,
                                                   FloatType delta_t) {

      p.set_x(p.get_x() + p.get_px() / p.get_mass() * 0.5 * delta_t);
      p.set_y(p.get_y() + p.get_py() / p.get_mass() * 0.5 * delta_t);
      p.set_z(p.get_z() + p.get_pz() / p.get_mass() * 0.5 * delta_t);
      p.set_t(p.get_t() + p.get_e() / p.get_mass() * 0.5 * delta_t);
    }

    /// Core function to integrate the momenta and determine the new position of
    /// a particle
    template <class Proxy, class Force, class FloatType>
    __saga_core_function__ void
    integrate_momenta_and_position(Proxy particle, Force const &force,
                                   FloatType delta_t) {

      auto mass =
          particle.get_mass(); // whatever we do, we must preserve the mass

      // momenta
      particle.set_momenta_and_mass(particle.get_px() + force.get_x() * delta_t,
                                    particle.get_py() + force.get_y() * delta_t,
                                    particle.get_pz() + force.get_z() * delta_t,
                                    mass);

      // integrate the positions
      detail::integrate_position(particle, delta_t);
    }
  } // namespace detail

  /// Functor that integrates the positions
  template <backend Backend> struct integrate_position;

  template <> struct integrate_position<backend::CPU> {
    template <class Particles, class FloatType>
    static void evaluate(Particles &particles, FloatType delta_t) {
      for (auto p : particles)
        detail::integrate_position(p, delta_t);
    }
  };

  template <> struct integrate_position<backend::CUDA> {
    template <class Particles, class FloatType>
    static void evaluate(Particles &particles, FloatType delta_t) {
#if SAGA_CUDA_ENABLED
      saga::core::cuda::apply_simple_function_inplace(
          particles, &detail::integrate_position, delta_t);
#else
      throw std::runtime_error("Attempt to call a method for the CUDA backend "
                               "when it is not enabled");
#endif
    }
  };

  /// Functor that fills an array of forces taking into account point-to-point
  /// interactions
  template <backend Backend> struct fill_forces;

  template <> struct fill_forces<backend::CPU> {

    template <class Forces, class Function, class Particles>
    static void evaluate(Forces &forces, Function &&function,
                         Particles const &particles) {

      for (auto i = 0u; i < particles.size(); ++i) {

        auto pi = particles[i];
        auto force_i = forces[i];

        for (auto j = i + 1; j < particles.size(); ++j) {

          auto pj = particles[j];
          auto force_j = forces[j];

          auto res = function(pi, pj);

          force_i.set_x(force_i.get_x() + res.get_x());
          force_i.set_y(force_i.get_y() + res.get_y());
          force_i.set_z(force_i.get_z() + res.get_z());

          force_j.set_x(force_j.get_x() - res.get_x());
          force_j.set_y(force_j.get_y() - res.get_y());
          force_j.set_z(force_j.get_z() - res.get_z());
        }
      }
    }
  };

  // TODO: Get the maximum number of threads per block from CUDA and allow to
  // pass the number of tiles as a configuration parameter. This configuration
  // could be specified during the construction of the saga::world object and
  // remain constant throughout the execution of the processes.
  template <> struct fill_forces<backend::CUDA> {

    template <class Forces, class Function, class Particles>
    static void evaluate(Forces &forces, Function &&function,
                         Particles const &particles) {

#if SAGA_CUDA_ENABLED
      auto N = particles.size();
      auto nblocks = N / SAGA_CUDA_MAX_THREADS_PER_BLOCK_X +
                     (N % SAGA_CUDA_MAX_THREADS_PER_BLOCK_X != 0);
      auto tile_size = N / 10; // TODO: this can be tunned
      auto smem = SAGA_CUDA_MAX_THREADS_PER_BLOCK_X *
                  sizeof(typename Particles::value_type);

      saga::core::cuda::
          add_forces<<<SAGA_CUDA_MAX_THREADS_PER_BLOCK_X, nblocks, smem>>>(
              tile_size, forces, force_function, particles);
#else
      throw std::runtime_error("Attempt to call a method for the CUDA backend "
                               "when it is not enabled");
#endif
    }
  };

  /// Functor to integrate the momenta and positions (simultaneously)
  template <backend Backend> struct integrate_momenta_and_position;

  template <> struct integrate_momenta_and_position<backend::CPU> {

    template <class Particles, class Forces, class FloatType>
    static void evaluate(Particles &particles, Forces const &forces,
                         FloatType delta_t) {

      for (auto i = 0u; i < particles.size(); ++i) {
        detail::integrate_momenta_and_position(particles[i], forces[i],
                                               delta_t);
      }
    }
  };

  template <> struct integrate_momenta_and_position<backend::CUDA> {

    template <class Particles, class Forces, class FloatType>
    static void evaluate(Particles &particles, Forces const &forces,
                         FloatType delta_t) {

#if SAGA_CUDA_ENABLED
      auto N = particles.size();
      auto nblocks = N / SAGA_CUDA_MAX_THREADS_PER_BLOCK_X +
                     (N % SAGA_CUDA_MAX_THREADS_PER_BLOCK_X != 0);
      saga::core::cuda::apply_contiguous_function_inplace(
          particles, forces, &detail::integrate_momenta_and_position, delta_t);
#else
      throw std::runtime_error("Attempt to call a method for the CUDA backend "
                               "when it is not enabled");
#endif
    }
  };
} // namespace saga::core
