#pragma once
#include "saga/core/types.hpp"
#include "saga/physics/charge.hpp"
#include "saga/physics/force.hpp"
#include "saga/physics/quantities.hpp"

#include <cmath>

namespace saga::physics {

  /*!\brief  Base class to define an interaction
   */
  template <class TypeDescriptor, template <class> class Charge, class Output,
            template <class> class... Property>
  struct interaction {

    /// Charge type
    using charge_type = Charge<TypeDescriptor>;

    /// Functor accessing the charge
    static constexpr charge_type charge = charge_type{};

    /// Evaluate the force for two objects
    template <class Proxy>
    Output operator()(Proxy const &src, Proxy const &tgt) const {
      return force(charge(src),
                   Property<TypeDescriptor>::template proxy_type<
                       typename Proxy::container_type>::get(src)...,
                   charge(tgt),
                   Property<TypeDescriptor>::template proxy_type<
                       typename Proxy::container_type>::get(tgt)...);
    }

    /// Evaluate the force given the two sets of properties
    virtual Output force(
        typename charge_type::float_type,
        typename Property<TypeDescriptor>::underlying_value_type...,
        typename charge_type::float_type,
        typename Property<TypeDescriptor>::underlying_value_type...) const = 0;
  };

  /*!\brief Represent any kind of central force working in the non-relativistic
   * regime
   */
  template <class TypeDescriptor, template <class> class Charge>
  struct central_force_non_relativistic
      : public saga::physics::interaction<
            TypeDescriptor, Charge,
            typename saga::physics::forces<TypeDescriptor>::value_type,
            property::x, property::y, property::z> {

    /// Base class
    using base_type = saga::physics::interaction<
        TypeDescriptor, Charge,
        typename saga::physics::forces<TypeDescriptor>::value_type, property::x,
        property::y, property::z, property::px, property::py, property::pz,
        property::e>;

    /// Floating-point type
    using float_type = typename TypeDescriptor::float_type;
    /// Value returned by the functor
    using return_type =
        typename saga::physics::forces<TypeDescriptor>::value_type;

    /// Constructor from the field constant
    central_force_non_relativistic(float_type k) : m_field_constant{k} {}

    /// Evaluate the force
    return_type force(float_t tgt_mass, float_type tgt_x, float_type tgt_y,
                      float_type tgt_z, float_type src_mass, float_type src_x,
                      float_type src_y, float_type src_z) const override {

      float_type const dx = src_x - tgt_x;
      float_type const dy = src_y - tgt_y;
      float_type const dz = src_z - tgt_z;

      float_type const r2 = dx * dx + dy * dy + dz * dz;

      if (r2 <= saga::types::numeric_info<TypeDescriptor>::min)
        return {0.f, 0.f, 0.f};

      float_type const tgt_force = m_field_constant * tgt_mass * src_mass / r2;

      float_type const r = std::sqrt(r2);

      float_type const ux = dx / r;
      float_type const uy = dy / r;
      float_type const uz = dz / r;

      return {tgt_force * ux, tgt_force * uy, tgt_force * uz};
    }

    /// Gravitational constant
    float_type m_field_constant = 0.f;
  };

} // namespace saga::physics
