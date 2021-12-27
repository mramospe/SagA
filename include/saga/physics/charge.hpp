#pragma once
#include "saga/core/types.hpp"
#include "saga/physics/quantities.hpp"

namespace saga::physics {

  namespace charge {

    /// Charge for gravitational interactions
    template <class TypeDescriptor> struct mass {
      /// Floating-point type
      using float_type = typename TypeDescriptor::float_type;
      /// Retrieve the mass
      template <class Proxy>
      __saga_core_function__ float_type operator()(Proxy const &p) const {
        return p.get_mass();
      }
    };

    /// Charge for gravitational interactions
    template <class TypeDescriptor> struct electric {
      /// Floating-point type
      using float_type = typename TypeDescriptor::float_type;
      /// Retrieve the mass
      template <class Proxy>
      constexpr __saga_core_function__ float_type
      operator()(Proxy const &p) const {
        return p.template get<saga::property::electric_charge>();
      }
    };
  } // namespace charge
} // namespace saga::physics
