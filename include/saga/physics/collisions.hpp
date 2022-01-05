#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/loops.hpp"
#include "saga/core/types.hpp"
#include "saga/core/vector.hpp"
#include "saga/physics/shape.hpp"
#include <algorithm>
#include <stdexcept>
#include <type_traits>

/// Elastic and inelastic collisions of particles
namespace saga::physics::collision {

  /*!\brief Helper class to evaluate the time a collision happens

      This is done by solving a second order polynomial. Numbers are computed
     efficiently to avoid doing unnecessary calculations. An imaginary value of
     the square root argument means that there was no collision in time. If the
     roots are positive, this means the collision is ahead in time so no action
     must be taken.
   */
  template <class TypeDescriptor> class collision_time_evaluator {

  public:
    using float_type = typename TypeDescriptor::float_type;

    /// Build the class and compute intermediate quantities
    template <class Proxy>
    __saga_core_function__ collision_time_evaluator(Proxy const &src,
                                                    Proxy const &tgt) {

      static_assert(std::is_same_v<typename Proxy::shape_type,
                                   saga::physics::sphere<TypeDescriptor>>);

      float_type dx = tgt.get_x() - src.get_x();
      float_type dy = tgt.get_y() - src.get_y();
      float_type dz = tgt.get_z() - src.get_z();

      float_type dpx = tgt.get_px() - src.get_px();
      float_type dpy = tgt.get_py() - src.get_py();
      float_type dpz = tgt.get_pz() - src.get_pz();

      float_type R = src.template get<saga::physics::radius>() +
                     tgt.template get<saga::physics::radius>();

      float_type d2 = dx * dx + dy * dy + dz * dz;

      float_type a = dpx * dpx + dpy * dpy + dpz * dpz;
      float_type b = 2 * (dx * dpx + dy * dpy + dz * dpz);
      float_type c = d2 - R * R;

      float_type sqrt_arg = b * b - 4 * a * c;

      // cases where particles are at rest
      if (std::abs(a) <= saga::numeric_info<TypeDescriptor>::min) {
        m_dt = saga::numeric_info<TypeDescriptor>::max;
      } else {

        if (sqrt_arg < 0)
          m_dt = saga::numeric_info<TypeDescriptor>::max;
        else if (sqrt_arg <= saga::numeric_info<TypeDescriptor>::min)
          m_dt = -0.5 * b / a;
        else {

          float_type t1 = 0.5 * (-b + std::sqrt(sqrt_arg)) / a;
          float_type t2 = 0.5 * (-b - std::sqrt(sqrt_arg)) / a;

          m_dt = t1 < t2 ? t1 : t2;
        }
      }
    }

    /// Whether there was a collision given the linear trayectories of the
    /// particles
    __saga_core_function__ bool has_collision() const { return (m_dt < 0); }

    /// Calculate the delta-time of a collision
    __saga_core_function__ float_type dt() const { return m_dt; }

  protected:
    /// Delta-time to the collision
    float_type m_dt;
  };

  /*!\brief Elastic collisions of particles
   */
  template <class TypeDescriptor> struct elastic {

    using float_type = typename TypeDescriptor::float_type;

    /// Evaluate the collisions among particles inplace
    template <class Particles>
    void operator()(Particles &particles, float_type delta_t) const {

      auto size = particles.size();

      saga::vector<bool, TypeDescriptor::backend> invalid(size);
      saga::core::set_vector_values<TypeDescriptor::backend>::evaluate(invalid,
                                                                       false);

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
    template <class Proxy>
    std::enable_if_t<std::is_same_v<typename Proxy::shape_type,
                                    saga::physics::sphere<TypeDescriptor>>,
                     bool>
    operator()(Proxy &src, Proxy &tgt, float_type delta_t) const {

      collision_time_evaluator<TypeDescriptor> const dce(src, tgt);

      // only real numbers represent collisions
      if (dce.has_collision() && dce.dt() > -delta_t) {

        // positions of the collision
        src.set_x(src.get_x() + src.get_px() * dce.dt());
        src.set_y(src.get_y() + src.get_py() * dce.dt());
        src.set_z(src.get_z() + src.get_pz() * dce.dt());

        tgt.set_x(tgt.get_x() + tgt.get_px() * dce.dt());
        tgt.set_y(tgt.get_y() + tgt.get_py() * dce.dt());
        tgt.set_z(tgt.get_z() + tgt.get_pz() * dce.dt());

        // we must recalculate the distances, this time at the collision point
        float_type dx_p = tgt.get_x() - src.get_x();
        float_type dy_p = tgt.get_y() - src.get_y();
        float_type dz_p = tgt.get_z() - src.get_z();

        float_type dpx_p = tgt.get_px() - src.get_px();
        float_type dpy_p = tgt.get_py() - src.get_py();
        float_type dpz_p = tgt.get_pz() - src.get_pz();

        float_type mt = 2.f * (dx_p * dpx_p + dy_p * dpy_p + dz_p * dpz_p) /
                        ((dx_p * dx_p + dy_p * dy_p + dz_p * dz_p) *
                         (src.get_mass() + tgt.get_mass()));

        float_type base_x = mt * dx_p;
        float_type base_y = mt * dy_p;
        float_type base_z = mt * dz_p;

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
        auto integrate_position = [&delta_t, &dce](auto &p) -> void {
          auto t = delta_t + dce.dt(); // dt is negative
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
  template <class TypeDescriptor> struct simple_merge {

    using float_type = typename TypeDescriptor::float_type;

    using merged_status = bool;
    static constexpr merged_status merged_status_true = true;
    static constexpr merged_status merged_status_false = false;

    /// Evaluate the collisions among particles inplace
    template <class Particles>
    void operator()(Particles &particles, float_type delta_t) const {

      auto size = particles.size();

      saga::vector<merged_status, TypeDescriptor::backend> invalid(size);
      saga::core::set_vector_values<TypeDescriptor::backend>::evaluate(invalid,
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
    template <class Proxy>
    void operator()(Proxy &src, Proxy &tgt, float_type delta_t) const {
      merge_if_close_and_return_status(src, tgt, delta_t);
    }

  private:
    /// Merge two particles if they are close enough, and return the
    /// corresponding status code
    template <class Proxy>
    std::enable_if_t<std::is_same_v<typename Proxy::shape_type,
                                    saga::physics::sphere<TypeDescriptor>>,
                     merged_status>
    merge_if_close_and_return_status(Proxy &src, Proxy &tgt,
                                     float_type delta_t) const {

      collision_time_evaluator<TypeDescriptor> const dce(src, tgt);

      // only real numbers represent collisions
      if (dce.has_collision() && dce.dt() > -delta_t) {

        auto radius_from_mass = [](auto const &p1_radius, auto const &p1_mass,
                                   auto const &p2_mass) {
          return std::pow(float_type{1.f} + p2_mass / p1_mass,
                          float_type{1.f} / float_type{3.f}) *
                 p1_radius;
        };

        auto src_mass = src.get_mass();
        auto tgt_mass = tgt.get_mass();

        float_type R =
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
        auto dt = -dce.dt(); // dt is negative
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
