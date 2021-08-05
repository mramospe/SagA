#pragma once
#include "saga/physics/electromagnetic.hpp"
#include "saga/physics/gravity.hpp"
#include <variant>
#include <vector>

/// Definition of physic parameters and interactions
namespace saga::physics {
  /// Represent a variant for any kind of interaction
  template <class TypeDescriptor>
  using interaction_variant =
      std::variant<
    //saga::physics::coulomb_non_relativistic_interaction<
    //		     TypeDescriptor>,
		   saga::physics::gravitational_non_relativistic_interaction<
          TypeDescriptor>>;

  /// Represent a collection of interactions
  template <class TypeDescriptor>
  using interactions_variant = std::vector<interaction_variant<TypeDescriptor>>;
} // namespace saga::physics
