#pragma once
#include "saga/physics/interaction.hpp"
#include "saga/physics/charge.hpp"

namespace saga::physics {
  /*!\brief Coulomb non-relativistic interaction
   */
  template <class TypeDescriptor>
  struct coulomb_non_relativistic_interaction
      : public saga::physics::central_force_non_relativistic<
            TypeDescriptor, saga::physics::charge::electric> {
  public:
    /// Base class
    using base_type = saga::physics::central_force_non_relativistic<
        TypeDescriptor, saga::physics::charge::electric>;
    /// Constructors inherited
    using base_type::base_type;
  };
} // namespace saga::physics
