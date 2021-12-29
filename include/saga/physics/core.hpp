#pragma once
#include "saga/physics/electromagnetic.hpp"
#include "saga/physics/gravity.hpp"
#include "saga/physics/quantities.hpp"
#include <type_traits>
#include <variant>
#include <vector>

/// Definition of physic parameters and interactions
namespace saga::physics {

  /// Check if the given interaction is valid for that set of properties
  template <template <class> class Interaction, class Properties>
  struct is_available_interaction : std::false_type {};

  /// Enable only
  template <template <class> class... Property>
  struct is_available_interaction<
      saga::physics::coulomb_non_relativistic_interaction,
      saga::properties<Property...>>
      : std::conditional_t<saga::core::has_single_template_v<
                               saga::property::electric_charge, Property...>,
                           std::true_type, std::false_type> {};

  /// Gravity can always be included
  template <template <class> class... Property>
  struct is_available_interaction<
      saga::physics::gravitational_non_relativistic_interaction,
      saga::properties<Property...>> : std::true_type {};

  /// Alias to check if the interaction is valid for the given properties
  template <template <class> class Interaction, class Properties>
  static constexpr bool is_available_interaction_v =
      is_available_interaction<Interaction, Properties>::value;

  /// Implementations that are specific to the physics namespace
  namespace detail {

    /// Represent a set of interactions
    template <template <class> class... Interaction> struct interactions {};

    /// All the existing interactions
    using all_interactions =
        interactions<saga::physics::gravitational_non_relativistic_interaction,
                     saga::physics::coulomb_non_relativistic_interaction>;

    /// Extend a variant of interactions if that given as template argument can
    /// be used for the given set of properties
    template <class Variant, class TypeDescriptor,
              template <class> class Interaction, class Properties>
    struct extend_variant_for_interaction;

    /// Extend a variant of interactions if that given as template argument can
    /// be used for the given set of properties
    template <class TypeDescriptor, class... T,
              template <class> class Interaction, class Properties>
    struct extend_variant_for_interaction<std::variant<T...>, TypeDescriptor,
                                          Interaction, Properties> {
      using type = std::conditional_t<
          is_available_interaction_v<Interaction, Properties>,
          std::variant<T..., Interaction<TypeDescriptor>>, std::variant<T...>>;
    };

    /// Extend a variant of interactions if that given as template argument can
    /// be used for the given set of properties
    template <class Variant, class TypeDescriptor,
              template <class> class Interaction, class Properties>
    using extend_variant_for_interaction_t =
        typename extend_variant_for_interaction<Variant, TypeDescriptor,
                                                Interaction, Properties>::type;

    /// Determine and represent a variant of interactions for the given set of
    /// properties
    template <class TypeDescriptor, class Variant, class Interactions,
              class Properties>
    struct interaction_variant_impl;

    /// Determine and represent a variant of interactions for the given set of
    /// properties
    template <class TypeDescriptor, class... T, template <class> class I0,
              template <class> class... Interaction, class Properties>
    struct interaction_variant_impl<TypeDescriptor, std::variant<T...>,
                                    interactions<I0, Interaction...>,
                                    Properties> {
      using type = typename interaction_variant_impl<
          TypeDescriptor,
          extend_variant_for_interaction_t<std::variant<T...>, TypeDescriptor,
                                           I0, Properties>,
          interactions<Interaction...>, Properties>::type;
    };

    /// Determine and represent a variant of interactions for the given set of
    /// properties
    template <class TypeDescriptor, class... T, template <class> class I0,
              class Properties>
    struct interaction_variant_impl<TypeDescriptor, std::variant<T...>,
                                    interactions<I0>, Properties> {
      using type =
          extend_variant_for_interaction_t<std::variant<T...>, TypeDescriptor,
                                           I0, Properties>;
    };
  } // namespace detail

  /// Represent a variant for any kind of interaction
  template <class TypeDescriptor, class Properties>
  using interaction_variant =
      typename detail::interaction_variant_impl<TypeDescriptor, std::variant<>,
                                                detail::all_interactions,
                                                Properties>::type;

  /// Represent a collection of interactions
  template <class TypeDescriptor, class Properties>
  using interactions_variant =
      std::vector<interaction_variant<TypeDescriptor, Properties>>;
} // namespace saga::physics
