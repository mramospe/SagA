#pragma once
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
    collision_time_evaluator(Proxy const &src, Proxy const &tgt) {

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

      m_a = dpx * dpx + dpy * dpy + dpz * dpz;
      m_b = 2 * (dx * dpx + dy * dpy + dz * dpz);
      float_type c = d2 - R * R;

      m_sqrt_arg = m_b * m_b - 4 * m_a * c;
    }

    /// Whether there was a collision given the linear trayectories of the
    /// particles
    bool has_collision() const { return (m_sqrt_arg >= 0); }

    /// Calculate the delta-time of a collision
    float_type dt() const {

      float_type t1 = (-m_b + std::sqrt(m_sqrt_arg)) / (2 * m_a);

      float_type dt = t1 < 0 ? t1 : (-m_b - std::sqrt(m_sqrt_arg)) / (2 * m_a);

      if (dt <= 0)
        throw std::runtime_error("Unexpected delta-time smaller than zero");

      return dt;
    }

  protected:
    /// Argument to the square root
    float_type m_sqrt_arg;
    /// Second order polynomial parameter "a"
    float_type m_a;
    /// Second order polynomial parameter "b"
    float_type m_b;
  };

  /*!\brief Elastic collisions of particles
   */
  template <class TypeDescriptor> struct elastic {

    using float_type = typename TypeDescriptor::float_type;

    /// Evaluate the collisions among particles inplace
    template <class Particles>
    void operator()(float_type delta_t, Particles &particles) const {

      auto size = particles.size();

      for (auto i = 0u; i < size; ++i) {

        auto pi = particles[i];

        for (auto j = i; j < size; ++j) {

          auto pj = particles[j];

          this->operator()(delta_t, pi, pj);
        }
      }
    }

    /// Evaluate the collisions among two particles
    template <class Proxy>
    std::enable_if_t<std::is_same_v<typename Proxy::shape_type,
                                    saga::physics::sphere<TypeDescriptor>>>
    operator()(float_type delta_t, Proxy &src, Proxy &tgt) const {

      collision_time_evaluator<TypeDescriptor> const dce(src, tgt);

      // only real numbers represent collisions
      if (dce.has_collision()) {

        float_type dt = dce.dt();

        if (dt > -delta_t) {

          // positions of the collision
          src.set_x(src.get_x() + src.get_px() * dt);
          src.set_y(src.get_y() + src.get_py() * dt);
          src.set_z(src.get_z() + src.get_pz() * dt);

          tgt.set_x(tgt.get_x() + tgt.get_px() * dt);
          tgt.set_y(tgt.get_y() + tgt.get_py() * dt);
          tgt.set_z(tgt.get_z() + tgt.get_pz() * dt);

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

          src.set_px(src.get_px() + base_x * tgt_mass);
          src.set_py(src.get_py() + base_y * tgt_mass);
          src.set_pz(src.get_pz() + base_z * tgt_mass);
          src.set_mass(src_mass);

          tgt.set_px(tgt.get_px() + base_x * src_mass);
          tgt.set_py(tgt.get_py() + base_y * src_mass);
          tgt.set_pz(tgt.get_pz() + base_z * src_mass);
          tgt.set_mass(tgt_mass);

          // integrate the positions for the time lapse since the collision
          auto integrate_position = [&delta_t, &dt](auto &p) -> void {
            auto t = delta_t + dt; // dt is negative
            p.set_x(p.get_x() + p.get_px() / p.get_mass() * t);
            p.set_y(p.get_y() + p.get_py() / p.get_mass() * t);
            p.set_z(p.get_z() + p.get_pz() / p.get_mass() * t);
            p.set_t(p.get_t() + p.get_e() / p.get_mass() * t);
          };

          integrate_position(src);
          integrate_position(tgt);
        }
      }
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
    void operator()(float_type delta_t, Particles &particles) const {

      auto size = particles.size();

      std::vector<merged_status> invalid(size, false);

      for (auto i = 0u; i < size; ++i) {

        if (invalid[i])
          continue;

        auto pi = particles[i];

        for (auto j = i + 1; j < size; ++j) {

          if (invalid[j])
            continue;

          auto pj = particles[j];

          if ((invalid[j] =
                   this->merge_if_close_and_return_status(delta_t, pi, pj)))
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

        particles[write_counter++] = std::move(particles[read_counter++]);
      }

      particles.resize(read_counter);
    }

    /// Evaluate the collisions among two particles
    template <class Proxy>
    void operator()(float_type delta_t, Proxy &src, Proxy &tgt) const {
      merge_if_close_and_return_status(delta_t, src, tgt);
    }

  private:
    /// Merge two particles if they are close enough, and return the
    /// corresponding status code
    template <class Proxy>
    std::enable_if_t<std::is_same_v<typename Proxy::shape_type,
                                    saga::physics::sphere<TypeDescriptor>>,
                     merged_status>
    merge_if_close_and_return_status(float_type delta_t, Proxy &src,
                                     Proxy &tgt) const {

      collision_time_evaluator<TypeDescriptor> const dce(src, tgt);

      // only real numbers represent collisions
      if (dce.has_collision()) {

        float_type dt = dce.dt();

        if (dt > -delta_t) {

          // positions of the collision
          src.set_x(0.5 * (src.get_x() + tgt.get_x()));
          src.set_x(0.5 * (src.get_y() + tgt.get_y()));
          src.set_x(0.5 * (src.get_z() + tgt.get_z()));

          src.set_px(src.get_px() + tgt.get_px());
          src.set_py(src.get_py() + tgt.get_py());
          src.set_pz(src.get_pz() + tgt.get_pz());
          src.set_mass(src.get_mass() + tgt.get_mass());

          // integrate the positions for the time lapse since the collision
          auto t = delta_t + dt; // dt is negative
          src.set_x(src.get_x() + src.get_px() / src.get_mass() * t);
          src.set_y(src.get_y() + src.get_py() / src.get_mass() * t);
          src.set_z(src.get_z() + src.get_pz() / src.get_mass() * t);
          src.set_t(src.get_t() + src.get_e() / src.get_mass() * t);

          // have merged
          return merged_status_true;
        }
      }
      // have not merged
      return merged_status_false;
    }
  };
} // namespace saga::physics::collision
