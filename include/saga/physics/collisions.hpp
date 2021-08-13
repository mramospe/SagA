#pragma once
#include "saga/physics/shape.hpp"
#include <type_traits>

namespace saga::physics::collision {

  template <class TypeDescriptor> struct elastic {

    using float_type = typename TypeDescriptor::float_type;

    template <class Particles>
    void operator()(float_type delta_t, Particles &particles) const {

      auto size = particles.size();

      for (auto i = 0u; i < size; ++i) {

        auto pi = particles[i];

        for (auto j = i; j < size; ++j) {

          auto pj = particles[j];

          this->operator()(pi, pj);
        }
      }
    }

    template <class Proxy>
    std::enable_if_t<
        std::is_same_v<typename Proxy::shape_type, saga::physics::sphere>>
    operator()(float_type delta_t, Proxy &src, Proxy &tgt) const {

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

      // only real numbers represent collisions
      if (sqrt_arg >= 0) {

        float_type t1 = (-b + std::sqrt(sqrt_arg)) / (2 * a);
        float_type t2 = (-b - std::sqrt(sqrt_arg)) / (2 * a);

        float_type dt = t1 < 0 ? t1 : t2;

        assert(dt <= 0);

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
} // namespace saga::physics::collision
