#pragma once
#include "saga/core/force.hpp"
#include "saga/core/keywords.hpp"
#include "saga/core/types.hpp"
#include "saga/physics/charge.hpp"
#include "saga/physics/quantities.hpp"

#include <cmath>

namespace saga::physics {

  /// Keyword argument for the field constant of a central force
  struct field_constant : public saga::core::keywords::keyword_float {};

  /// Keyword argument for the the soften factor of a central force
  struct soften_factor : public saga::core::keywords::keyword_float {};

  /*!\brief  Base class to define an interaction
   */
  template <class TypeDescriptor, template <class> class Charge, class Output,
            template <class> class... Property>
  struct interaction {

    /// Charge type
    using charge_type = Charge<TypeDescriptor>;

    /// Evaluate the force for two objects
    template <class P0, class P1>
    __saga_core_function__ Output operator()(P0 const &src,
                                             P1 const &tgt) const {
      return force(charge_type{}(src), src.template get<Property>()...,
                   charge_type{}(tgt), tgt.template get<Property>()...);
    }

    /// Evaluate the force given the two sets of properties
    __saga_core_function__ virtual Output force(
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
            typename saga::core::forces<TypeDescriptor>::value_type,
            property::x, property::y, property::z>,
        public saga::core::keywords::keywords_parser<
            TypeDescriptor, saga::core::keywords::required<field_constant>,
            soften_factor> {

    using interaction_base_type = saga::physics::interaction<
        TypeDescriptor, Charge,
        typename saga::core::forces<TypeDescriptor>::value_type, property::x,
        property::y, property::z>;
    using keywords_parser_base_type = saga::core::keywords::keywords_parser<
        TypeDescriptor, saga::core::keywords::required<field_constant>,
        soften_factor>;

    /// Construction from keyword arguments
    template <class... K>
    central_force_non_relativistic(K &&...v) noexcept
        : interaction_base_type{},
          keywords_parser_base_type{
              saga::core::make_tuple(
                  soften_factor{saga::numeric_info<TypeDescriptor>::min}),
              std::forward<K>(v)...} {}

    // Allow copy/move ellision
    central_force_non_relativistic(central_force_non_relativistic const &) =
        default;
    central_force_non_relativistic(central_force_non_relativistic &&) = default;
    central_force_non_relativistic &
    operator=(central_force_non_relativistic const &) = default;
    central_force_non_relativistic &
    operator=(central_force_non_relativistic &&) = default;

    /// Floating-point type
    using float_type = typename TypeDescriptor::float_type;
    /// Value returned by the functor
    using return_type = typename saga::core::forces<TypeDescriptor>::value_type;

    /// Evaluate the force
    __saga_core_function__ return_type force(float_type tgt_charge,
                                             float_type tgt_x, float_type tgt_y,
                                             float_type tgt_z,
                                             float_type src_charge,
                                             float_type src_x, float_type src_y,
                                             float_type src_z) const override {

      float_type const dx = src_x - tgt_x;
      float_type const dy = src_y - tgt_y;
      float_type const dz = src_z - tgt_z;

      float_type const r2 =
          dx * dx + dy * dy + dz * dz + this->template get<soften_factor>();

      float_type const tgt_force =
          this->template get<field_constant>() * tgt_charge * src_charge / r2;

      float_type const r = std::sqrt(r2);

      float_type const ux = dx / r;
      float_type const uy = dy / r;
      float_type const uz = dz / r;

      return {tgt_force * ux, tgt_force * uy, tgt_force * uz};
    }
  };

} // namespace saga::physics
