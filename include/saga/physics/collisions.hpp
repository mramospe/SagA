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
    /*!\brief Helper class to evaluate the time a collision happens

      This is done by solving a second order polynomial. Numbers are computed
      efficiently to avoid doing unnecessary calculations. An imaginary value of
      the square root argument means that there was no collision in time. If the
      roots are positive, this means the collision is ahead in time so no action
      must be taken.
    */
    template <class Proxy>
    __saga_core_function__ typename Proxy::type_descriptor::float_type
    evaluate_collision_time(Proxy const &src, Proxy const &tgt) {

      using type_descriptor = typename Proxy::type_descriptor;

      static_assert(std::is_same_v<typename Proxy::shape_type,
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
      if (std::abs(a) <= saga::numeric_info<type_descriptor>::min) {
        return saga::numeric_info<type_descriptor>::max;
      } else {

        if (sqrt_arg < 0)
          return saga::numeric_info<type_descriptor>::max;
        else if (sqrt_arg <= saga::numeric_info<type_descriptor>::min)
          return -0.5 * b / a;
        else {

          auto t1 = 0.5 * (-b + std::sqrt(sqrt_arg)) / a;
          auto t2 = 0.5 * (-b - std::sqrt(sqrt_arg)) / a;

          return t1 < t2 ? t1 : t2;
        }
      }
    }

    /// Whether there was a collision given the linear trayectories of the
    /// particles
    template <class FloatType>
    __saga_core_function__ bool is_valid_collision(FloatType dt,
                                                   FloatType delta_t) {
      return dt < 0 && dt > -delta_t;
    }
  } // namespace detail

  /*!\brief Elastic collisions of particles
   */
  template <saga::backend Backend> struct elastic {

    /// Evaluate the collisions among particles inplace
    template <class Particles, class FloatType>
    void operator()(Particles &particles, FloatType delta_t) const {

      auto size = particles.size();

      saga::core::vector<bool, Backend> invalid(size);
      saga::physics::set_vector_values<Backend>::evaluate(invalid, false);

      for (auto i = 0u; i < size; ++i) {

        auto pi = particles[i];

        for (auto j = i + 1; j < size; ++j) {

          if (invalid[j])
            continue;

          auto pj = particles[j];

          if (this->operator()(pi, pj, delta_t)) {
            // no need to set/check invalid[i] since we will never end-up
            // processing that particle again
            invalid[j] = true;
            break;
          }
        }
      }
    }

    /// Evaluate the collisions between two particles
    template <class Proxy, class FloatType>
    std::enable_if_t<
        std::is_same_v<typename Proxy::shape_type,
                       saga::physics::sphere<typename Proxy::type_descriptor>>,
        bool>
    operator()(Proxy &src, Proxy &tgt, FloatType delta_t) const {

      auto time_to_collision = detail::evaluate_collision_time(src, tgt);

      // only real numbers represent collisions
      if (detail::is_valid_collision(time_to_collision, delta_t)) {

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

        return true;
      } else
        return false;
    }
  };

  /*!\brief Collision type where two balls merge into one when are too close
   */
  template <saga::backend Backend> struct simple_merge {

    using merged_status = bool;
    static constexpr merged_status merged_status_true = true;
    static constexpr merged_status merged_status_false = false;

    /// Evaluate the collisions among particles inplace
    template <class Particles, class FloatType>
    void operator()(Particles &particles, FloatType delta_t) const {

      auto size = particles.size();

      saga::core::vector<merged_status, Backend> invalid(size);
      saga::physics::set_vector_values<Backend>::evaluate(invalid, false);

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
          if ((invalid[j] =
                   this->merge_if_close_and_return_status(pi, pj, delta_t)))
            break;
        }
      }

      std::size_t read_counter = 0u;
      std::size_t write_counter = 0u;

      while (read_counter < size) {

        if (invalid[read_counter]) {
          ++read_counter;
          continue;
        }

        particles[write_counter++] = particles[read_counter++];
      }

      particles.resize(write_counter);
    }

    /// Evaluate the collisions among two particles
    template <class Proxy, class FloatType>
    void operator()(Proxy &src, Proxy &tgt, FloatType delta_t) const {
      merge_if_close_and_return_status(src, tgt, delta_t);
    }

  private:
    /// Merge two particles if they are close enough, and return the
    /// corresponding status code
    template <class Proxy, class FloatType>
    std::enable_if_t<
        std::is_same_v<typename Proxy::shape_type,
                       saga::physics::sphere<typename Proxy::type_descriptor>>,
        merged_status>
    merge_if_close_and_return_status(Proxy &src, Proxy &tgt,
                                     FloatType delta_t) const {

      auto time_to_collision = detail::evaluate_collision_time(src, tgt);

      // only real numbers represent collisions
      if (detail::is_valid_collision(time_to_collision, delta_t)) {

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

        // have merged
        return merged_status_true;
      }
      // have not merged
      return merged_status_false;
    }
  };
} // namespace saga::physics::collision
