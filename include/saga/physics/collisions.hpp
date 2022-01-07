#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/vector.hpp"
#include "saga/physics/loops.hpp"
#include "saga/physics/shape.hpp"
#include <algorithm>
#include <stdexcept>
#include <type_traits>

/// Elastic and inelastic collisions of particles
namespace saga::physics::collision {

  namespace detail {

    template <class FloatType> struct collision_information {
      FloatType delta_t;
      bool is_valid;
    };

    /*!\brief Helper class to evaluate the time a collision happens

      This is done by solving a second order polynomial. Numbers are computed
      efficiently to avoid doing unnecessary calculations. An imaginary value of
      the square root argument means that there was no collision in time. If the
      roots are positive, this means the collision is ahead in time so no action
      must be taken.
    */
    struct collision_time_evaluator {

      template <class U, class V>
      __saga_core_function__
          collision_information<typename U::type_descriptor::float_type>
          operator()(U const src, V const tgt,
                     typename U::type_descriptor::float_type delta_t) {

        using type_descriptor = typename U::type_descriptor;

        static_assert(std::is_same_v<typename U::shape_type,
                                     saga::physics::sphere<type_descriptor>>);

        auto dx = tgt.get_x() - src.get_x();
        auto dy = tgt.get_y() - src.get_y();
        auto dz = tgt.get_z() - src.get_z();

        auto dpx = tgt.get_px() - src.get_px();
        auto dpy = tgt.get_py() - src.get_py();
        auto dpz = tgt.get_pz() - src.get_pz();

        auto R = src.template get<saga::physics::radius>() +
                 tgt.template get<saga::physics::radius>();

        auto d2 = dx * dx + dy * dy + dz * dz;

        auto a = dpx * dpx + dpy * dpy + dpz * dpz;
        auto b = 2 * (dx * dpx + dy * dpy + dz * dpz);
        auto c = d2 - R * R;

        auto sqrt_arg = b * b - 4 * a * c;

        // cases where particles are at rest
        typename U::type_descriptor::float_type dt;
        if (std::abs(a) <= saga::numeric_info<type_descriptor>::min) {
          dt = saga::numeric_info<type_descriptor>::max;
        } else {

          if (sqrt_arg < 0)
            dt = saga::numeric_info<type_descriptor>::max;
          else if (sqrt_arg <= saga::numeric_info<type_descriptor>::min)
            dt = -0.5 * b / a;
          else {

            auto t1 = 0.5 * (-b + std::sqrt(sqrt_arg)) / a;
            auto t2 = 0.5 * (-b - std::sqrt(sqrt_arg)) / a;

            dt = t1 < t2 ? t1 : t2;
          }
        }

        return {dt, dt < 0 && dt > -delta_t};
      }
    };

    struct elastic_fctr {
      /*!\brief Evaluate the collisions between two particles

        Note that *time_to_collision* must be negative and smaller than
        *delta_t* in absolute value.
      */
      template <class U, class V, class FloatType>
      __saga_core_function__ std::enable_if_t<
          std::is_same_v<typename U::shape_type,
                         saga::physics::sphere<typename U::type_descriptor>>,
          void>
      operator()(U src, V tgt, FloatType time_to_collision,
                 FloatType delta_t) const {

        // positions of the collision
        src.set_x(src.get_x() + src.get_px() * time_to_collision);
        src.set_y(src.get_y() + src.get_py() * time_to_collision);
        src.set_z(src.get_z() + src.get_pz() * time_to_collision);

        tgt.set_x(tgt.get_x() + tgt.get_px() * time_to_collision);
        tgt.set_y(tgt.get_y() + tgt.get_py() * time_to_collision);
        tgt.set_z(tgt.get_z() + tgt.get_pz() * time_to_collision);

        // we must recalculate the distances, this time at the collision point
        auto dx_p = tgt.get_x() - src.get_x();
        auto dy_p = tgt.get_y() - src.get_y();
        auto dz_p = tgt.get_z() - src.get_z();

        auto dpx_p = tgt.get_px() - src.get_px();
        auto dpy_p = tgt.get_py() - src.get_py();
        auto dpz_p = tgt.get_pz() - src.get_pz();

        auto mt = 2.f * (dx_p * dpx_p + dy_p * dpy_p + dz_p * dpz_p) /
                  ((dx_p * dx_p + dy_p * dy_p + dz_p * dz_p) *
                   (src.get_mass() + tgt.get_mass()));

        auto base_x = mt * dx_p;
        auto base_y = mt * dy_p;
        auto base_z = mt * dz_p;

        // momenta
        auto src_mass = src.get_mass();
        auto tgt_mass = tgt.get_mass();

        src.set_momenta_and_mass(src.get_px() + base_x * tgt_mass,
                                 src.get_py() + base_y * tgt_mass,
                                 src.get_pz() + base_z * tgt_mass, src_mass);
        tgt.set_momenta_and_mass(tgt.get_px() + base_x * src_mass,
                                 tgt.get_py() + base_y * src_mass,
                                 tgt.get_pz() + base_z * src_mass, tgt_mass);

        // integrate the positions for the time lapse since the collision
        auto integrate_position = [&delta_t,
                                   &time_to_collision](auto &p) -> void {
          auto t = delta_t + time_to_collision; // time_to_collision is negative
          p.set_x(p.get_x() + p.get_px() / p.get_mass() * t);
          p.set_y(p.get_y() + p.get_py() / p.get_mass() * t);
          p.set_z(p.get_z() + p.get_pz() / p.get_mass() * t);
          p.set_t(p.get_t() + p.get_e() / p.get_mass() * t);
        };

        integrate_position(src);
        integrate_position(tgt);
      }
    };

    struct simple_merge_fctr {

      /// Merge two particles if they are close enough
      template <class U, class V, class FloatType>
      __saga_core_function__ std::enable_if_t<
          std::is_same_v<typename U::shape_type,
                         saga::physics::sphere<typename U::type_descriptor>>,
          void>
      operator()(U src, V tgt, FloatType time_to_collision,
                 FloatType delta_t) const {

        // only real numbers represent collisions
        auto radius_from_mass = [](auto const &p1_radius, auto const &p1_mass,
                                   auto const &p2_mass) {
          return std::pow(FloatType{1.f} + p2_mass / p1_mass,
                          FloatType{1.f} / FloatType{3.f}) *
                 p1_radius;
        };

        auto src_mass = src.get_mass();
        auto tgt_mass = tgt.get_mass();

        auto R =
            src_mass > tgt_mass
                ? radius_from_mass(src.template get<saga::physics::radius>(),
                                   src_mass, tgt_mass)
                : radius_from_mass(tgt.template get<saga::physics::radius>(),
                                   tgt_mass, src_mass);

        auto total_mass = src_mass + tgt_mass;

        // positions of the collision
        src.set_x((src_mass * src.get_x() + tgt_mass * tgt.get_x()) /
                  total_mass);
        src.set_y((src_mass * src.get_y() + tgt_mass * tgt.get_y()) /
                  total_mass);
        src.set_z((src_mass * src.get_z() + tgt_mass * tgt.get_z()) /
                  total_mass);

        src.set_px(src.get_px() + tgt.get_px());
        src.set_py(src.get_py() + tgt.get_py());
        src.set_pz(src.get_pz() + tgt.get_pz());
        src.set_e(src.get_e() + tgt.get_e());

        src.template set<saga::physics::radius>(R);

        // integrate the positions for the time lapse since the collision
        auto dt = -time_to_collision; // time_to_collision is negative
        src.set_x(src.get_x() + src.get_px() / src.get_mass() * dt);
        src.set_y(src.get_y() + src.get_py() / src.get_mass() * dt);
        src.set_z(src.get_z() + src.get_pz() / src.get_mass() * dt);
        src.set_t(src.get_t() + src.get_e() / src.get_mass() * dt);
      }
    };
  } // namespace detail

  /*!\brief Elastic collisions of particles
   */
  template <saga::backend Backend> struct elastic;

  template <> struct elastic<saga::backend::CPU> {

    /// Evaluate the collisions among particles inplace
    template <class Particles, class FloatType>
    void operator()(Particles &particles, FloatType delta_t) const {

      auto size = particles.size();

      saga::core::vector<bool, saga::backend::CPU> invalid(size);
      saga::physics::set_vector_values<saga::backend::CPU>::evaluate(invalid,
                                                                     false);

      for (auto i = 0u; i < size; ++i) {

        auto pi = particles[i];

        for (auto j = i + 1; j < size; ++j) {

          if (invalid[j])
            continue;

          auto pj = particles[j];

          auto [time_to_collision, is_valid] =
              detail::collision_time_evaluator{}(pi, pj, delta_t);

          if (is_valid) {

            detail::elastic_fctr{}(pi, pj, time_to_collision, delta_t);

            // no need to set/check invalid[i] since we will never end-up
            // processing that particle again
            invalid[j] = true;
            break;
          }
        }
      }
    }
  };

  template <> struct elastic<saga::backend::CUDA> {

    /// Evaluate the collisions among particles inplace
    template <class Particles, class FloatType>
    void operator()(Particles &particles, FloatType delta_t) const {

#if SAGA_CUDA_ENABLED

      auto [blocks, threads_per_block] =
          saga::core::cuda::optimal_grid_1d(particles);

      auto particles_view = saga::core::make_container_view(particles);

      auto smem = threads_per_block *
                  sizeof(typename decltype(particles_view)::value_type);

      saga::core::cuda::apply_functor_skip_if_previous_evaluation_is_true<<<
          blocks, threads_per_block, smem>>>(particles_view,
                                             detail::collision_time_evaluator{},
                                             detail::elastic_fctr{}, delta_t);

      SAGA_CHECK_LAS_ERROR("Failed to calculate collisions");
#else
      SAGA_THROW_CUDA_ERROR;
#endif
    }
  };

  /*!\brief Collision type where two balls merge into one when are too close
   */
  template <saga::backend Backend> struct simple_merge;

  template <> struct simple_merge<saga::backend::CPU> {

    /// Evaluate the collisions among particles inplace
    template <class Particles, class FloatType>
    void operator()(Particles &particles, FloatType delta_t) const {

      auto size = particles.size();

      saga::core::vector<bool, saga::backend::CPU> invalid(size);
      saga::physics::set_vector_values<saga::backend::CPU>::evaluate(invalid,
                                                                     false);

      for (auto i = 0u; i < size; ++i) {

        if (invalid[i])
          continue;

        auto pi = particles[i];

        for (auto j = i + 1; j < size; ++j) {

          if (invalid[j])
            continue;

          auto pj = particles[j];

          // if delta_t is too big, we might miss situations where several
          // particles collide at the same time
          auto [time_to_collision, is_valid] =
              detail::collision_time_evaluator{}(pi, pj, delta_t);

          if (is_valid) {
            detail::simple_merge_fctr{}(pi, pj, time_to_collision, delta_t);
            invalid[j] = true;
            break;
          }
        }
      }

      // allocate the new container of particles
      std::size_t n = 0;
      for (auto i = 0u; i < invalid.size(); ++i)
        if (!invalid[i])
          ++n;

      std::remove_reference_t<std::remove_cv_t<decltype(particles)>>
          new_particles(n);

      // copy the information to the new container
      std::size_t read_counter = 0u, write_counter = 0u;

      while (read_counter < size) {

        if (invalid[read_counter]) {
          ++read_counter;
          continue;
        }

        new_particles[write_counter++] = particles[read_counter++];
      }

      // set the input particles to the new particles (avoid a copy)
      particles = std::move(new_particles);
    }
  };

  template <> struct simple_merge<saga::backend::CUDA> {

    template <class Particles, class FloatType>
    void operator()(Particles &particles, FloatType delta_t) {

#if SAGA_CUDA_ENABLED

      auto [blocks, threads_per_block] =
          saga::core::cuda::optimal_grid_1d(particles);

      auto particles_view = saga::core::make_container_view(particles);

      auto smem = threads_per_block *
                  sizeof(typename decltype(particles_view)::value_type);

      saga::core::cuda::apply_functor_skip_if_previous_evaluation_is_true<<<
          blocks, threads_per_block, smem>>>(
          particles_view, detail::collision_time_evaluator{},
          detail::simple_merge_fctr{}, delta_t);

      // TODO: determine the new number of particles and allocate new vector

      SAGA_CHECK_LAS_ERROR("Failed to calculate collisions");
#else
      SAGA_THROW_CUDA_ERROR;
#endif
    }
  };
} // namespace saga::physics::collision
