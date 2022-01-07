#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/views.hpp"
#if SAGA_CUDA_ENABLED
#include "saga/cuda/core.hpp"
#include "saga/cuda/loops.hpp"
#endif

namespace saga::physics {

  namespace detail {

    /// Core function to integrate the position of a particle
    struct integrate_position_kernel_fctr {

      template <class Proxy, class FloatType>
      __saga_core_function__ void operator()(Proxy p, FloatType delta_t) const {

        p.set_x(p.get_x() + p.get_px() / p.get_mass() * 0.5 * delta_t);
        p.set_y(p.get_y() + p.get_py() / p.get_mass() * 0.5 * delta_t);
        p.set_z(p.get_z() + p.get_pz() / p.get_mass() * 0.5 * delta_t);
        p.set_t(p.get_t() + p.get_e() / p.get_mass() * 0.5 * delta_t);
      }
    };

    /// Core function to integrate the momenta and determine the new position of
    /// a particle
    struct integrate_momenta_and_position_kernel_fctr {

      template <class ParticleProxy, class ForceProxy, class FloatType>
      __saga_core_function__ void operator()(ParticleProxy particle,
                                             ForceProxy const force,
                                             FloatType delta_t) const {

        auto mass =
            particle.get_mass(); // whatever we do, we must preserve the mass

        // momenta
        particle.set_momenta_and_mass(
            particle.get_px() + force.get_x() * delta_t,
            particle.get_py() + force.get_y() * delta_t,
            particle.get_pz() + force.get_z() * delta_t, mass);

        // integrate the positions
        detail::integrate_position_kernel_fctr{}(particle, delta_t);
      }
    };
  } // namespace detail

#if SAGA_CUDA_ENABLED
  template <class View, class Functor, class... Args>
  void apply_functor(View obj, Functor const &functor, Args &&...args) {

    auto [blocks, threads_per_block] = saga::cuda::optimal_grid_1d(obj);

    saga::cuda::apply_functor<<<blocks, threads_per_block>>>(obj, functor,
                                                             args...);

    SAGA_CHECK_LAS_ERROR("Failed to evaluate functor");
  }
#endif

  template <backend Backend> struct set_vector_values;

  template <> struct set_vector_values<backend::CPU> {
    template <class Vector>
    static void evaluate(Vector &v, typename Vector::value_type def) {
      for (auto i = 0u; i < v.size(); ++i)
        v[i] = def;
    }
  };

  template <> struct set_vector_values<backend::CUDA> {
    template <class Vector>
    static void evaluate(Vector &v, typename Vector::value_type def) {
#if SAGA_CUDA_ENABLED

      auto [blocks, threads_per_block] = saga::cuda::optimal_grid_1d(v);

      saga::cuda::set_view_values<<<blocks, threads_per_block>>>(
          saga::core::make_vector_view(v), def);

      SAGA_CHECK_LAS_ERROR("Failed to set vector values");
#else
      SAGA_THROW_CUDA_ERROR;
#endif
    }
  };

  /// Functor that integrates the positions
  template <backend Backend> struct integrate_position;

  template <> struct integrate_position<backend::CPU> {
    template <class Particles, class FloatType>
    static void evaluate(Particles &particles, FloatType delta_t) {
      for (auto p : particles)
        detail::integrate_position_kernel_fctr{}(p, delta_t);
    }
  };

  template <> struct integrate_position<backend::CUDA> {
    template <class Particles, class FloatType>
    static void evaluate([[maybe_unused]] Particles &particles,
                         [[maybe_unused]] FloatType delta_t) {
#if SAGA_CUDA_ENABLED
      apply_functor(saga::core::make_container_view(particles),
                    detail::integrate_position_kernel_fctr{}, delta_t);
#else
      SAGA_THROW_CUDA_ERROR;
#endif
    }
  };

  /// Functor that fills an array of forces taking into account point-to-point
  /// interactions
  template <backend Backend> struct iterate_forces;

  template <> struct iterate_forces<backend::CPU> {

    template <class Forces> static void set_to_zero(Forces &forces) {
      for (auto f : forces)
        f = {0.f, 0.f, 0.f};
    }

    template <class Forces, class Functor, class Particles>
    static void fill_from_interaction(Forces &forces, Functor const &function,
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
  template <> struct iterate_forces<backend::CUDA> {

    template <class Forces>
    static void set_to_zero([[maybe_unused]] Forces &forces) {

#if SAGA_CUDA_ENABLED

      auto [blocks, threads_per_block] = saga::cuda::optimal_grid_1d(forces);

      auto forces_view = saga::core::make_container_view(forces);

      saga::cuda::set_view_values<<<blocks, threads_per_block>>>(
          forces_view,
          typename decltype(forces_view)::value_type{0.f, 0.f, 0.f});

      SAGA_CHECK_LAS_ERROR("Failed to set forces to zero");
#else
      SAGA_THROW_CUDA_ERROR;
#endif
    }

    template <class Forces, class Functor, class Particles>
    static void
    fill_from_interaction([[maybe_unused]] Forces &forces,
                          [[maybe_unused]] Functor const &force_function,
                          [[maybe_unused]] Particles const &particles) {

#if SAGA_CUDA_ENABLED

      auto [blocks, threads_per_block] = saga::cuda::optimal_grid_1d(particles);

      auto particles_view = saga::core::make_container_view(particles);
      auto forces_view = saga::core::make_container_view(forces);

      auto smem = threads_per_block *
                  sizeof(typename decltype(particles_view)::value_type);

      saga::cuda::calculate_forces<<<blocks, threads_per_block, smem>>>(
          forces_view, force_function, particles_view);

      SAGA_CHECK_LAS_ERROR("Failed to determine accelerations");
#else
      SAGA_THROW_CUDA_ERROR;
#endif
    }
  };

  /// Functor to integrate the momenta and positions (simultaneously)
  template <backend Backend> struct integrate_momenta_and_position;

  template <> struct integrate_momenta_and_position<backend::CPU> {

    template <class Particles, class Forces, class FloatType>
    static __saga_core_function__ void
    evaluate(Particles &particles, Forces const &forces, FloatType delta_t) {

      for (auto i = 0u; i < particles.size(); ++i) {
        detail::integrate_momenta_and_position_kernel_fctr{}(
            particles[i], forces[i], delta_t);
      }
    }
  };

  template <> struct integrate_momenta_and_position<backend::CUDA> {

    template <class Particles, class Forces, class FloatType>
    static void evaluate([[maybe_unused]] Particles &particles,
                         [[maybe_unused]] Forces const &forces,
                         [[maybe_unused]] FloatType delta_t) {

#if SAGA_CUDA_ENABLED

      auto [blocks, threads_per_block] = saga::cuda::optimal_grid_1d(particles);

      saga::cuda::apply_functor_contiguous_views<<<blocks, threads_per_block>>>(
          saga::core::make_container_view(particles),
          saga::core::make_container_view(forces),
          detail::integrate_momenta_and_position_kernel_fctr{}, delta_t);

      SAGA_CHECK_LAS_ERROR("Unable to integrate momenta and position");
#else
      SAGA_THROW_CUDA_ERROR;
#endif
    }
  };
} // namespace saga::physics
