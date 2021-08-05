#pragma once
#include "saga/physics/interaction.hpp"
#include "saga/physics/charge.hpp"

namespace saga::physics {
  /*!\brief Gravitational non-relativistic interaction
   */
  template <class TypeDescriptor>
  struct gravitational_non_relativistic_interaction
      : public saga::physics::central_force_non_relativistic<
            TypeDescriptor, saga::physics::charge::mass> {
  public:
    /// Base class
    using base_type = saga::physics::central_force_non_relativistic<
        TypeDescriptor, saga::physics::charge::mass>;
    /// Constructors inherited
    using base_type::base_type;
  };
} // namespace saga::physics
