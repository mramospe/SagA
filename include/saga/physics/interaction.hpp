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

  /*!\brief Represent any kind of central force working in the non-relativistic
   * regime
   */
  template <class TypeDescriptor, template <class> class Charge>
  struct central_force_non_relativistic
      : public saga::core::keywords::keywords_parser<
            TypeDescriptor, saga::core::keywords::required<field_constant>,
            soften_factor> {

    using keywords_parser_base_type = saga::core::keywords::keywords_parser<
        TypeDescriptor, saga::core::keywords::required<field_constant>,
        soften_factor>;

    /// Charge type
    using charge_type = Charge<TypeDescriptor>;
    /// Floating-point type
    using float_type = typename TypeDescriptor::float_type;
    /// Type of the value returned on call
    using return_type = typename saga::core::forces<TypeDescriptor>::value_type;

    /// Construction from keyword arguments
    template <class... K>
    central_force_non_relativistic(K &&...v)
        : keywords_parser_base_type{
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

    /// Evaluate the force given the two sets of properties
    template <class U, class V>
    __saga_core_function__ return_type operator()(U const &src,
                                                  V const &tgt) const {

      auto tgt_charge = charge_type{}(tgt);
      auto tgt_x = tgt.get_x();
      auto tgt_y = tgt.get_y();
      auto tgt_z = tgt.get_z();

      auto src_charge = charge_type{}(src);
      auto src_x = src.get_x();
      auto src_y = src.get_y();
      auto src_z = src.get_z();

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
