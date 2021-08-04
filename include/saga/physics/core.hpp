#pragma once
#include "saga/core/types.hpp"
#include "saga/physics/force.hpp"
#include "saga/physics/interaction.hpp"
#include "saga/physics/quantities.hpp"
#include <cmath>
#include <tuple>
#include <variant>
#include <vector>

namespace saga {

  /*!\brief Gravitational non-relativistic interaction
   */
  template <class TypeDescriptor>
  struct gravitational_non_relativistic_interaction
      : public saga::physics::interaction<
            TypeDescriptor,
            typename saga::physics::forces<TypeDescriptor>::value_type,
            property::x, property::y, property::z, property::px, property::py,
            property::pz, property::e> {

  public:
    /// Value returned by the functor
    using return_type =
        typename saga::physics::forces<TypeDescriptor>::value_type;

    /// Floating-point type to use
    using float_type = typename TypeDescriptor::float_type;

    /// Build from the interaction constants
    gravitational_non_relativistic_interaction() = default;
    gravitational_non_relativistic_interaction(float_type G)
        : m_gravitational_constant{G} {}

    /// Evaluate the force
    return_type force(float_type delta_t, float_type tgt_x, float_type tgt_y,
                      float_type tgt_z, float_type tgt_px, float_type tgt_py,
                      float_type tgt_pz, float_type tgt_e, float_type src_x,
                      float_type src_y, float_type src_z, float_type src_px,
                      float_type src_py, float_type src_pz,
                      float_type src_e) const override {

      float_type const tgt_mass = std::sqrt(std::abs(
          tgt_e * tgt_e - tgt_px * tgt_px - tgt_py * tgt_py - tgt_pz * tgt_pz));
      float_type const src_mass = std::sqrt(std::abs(
          src_e * src_e - src_px * src_px - src_py * src_py - src_pz * src_pz));

      float_type const dx = src_x - tgt_x;
      float_type const dy = src_y - tgt_y;
      float_type const dz = src_z - tgt_z;

      float_type const r2 = dx * dx + dy * dy + dz * dz;

      if (r2 <= saga::types::numeric_info<TypeDescriptor>::min)
        return {0.f, 0.f, 0.f, 0.f};

      float_type const tgt_force =
          m_gravitational_constant * tgt_mass * src_mass / r2;

      float_type const r = std::sqrt(r2);

      float_type const ux = dx / r;
      float_type const uy = dy / r;
      float_type const uz = dz / r;

      float_type const mom =
          std::sqrt(tgt_px * tgt_px + tgt_py * tgt_py + tgt_pz * tgt_pz);
      float_type const de_dt = mom * tgt_force / (tgt_mass * delta_t);

      return {tgt_force * ux, tgt_force * uy, tgt_force * uz, de_dt};
    }

    /// Gravitational constant
    float_type m_gravitational_constant = 6.67430e-11;
  };

  namespace physics {
    /// Represent a variant for any kind of interaction
    template <class TypeDescriptor>
    using interaction_variant = std::variant<
        gravitational_non_relativistic_interaction<TypeDescriptor>>;

    /// Represent a collection of interactions
    template <class TypeDescriptor>
    using interactions_variant =
        std::vector<interaction_variant<TypeDescriptor>>;
  } // namespace physics

} // namespace saga
