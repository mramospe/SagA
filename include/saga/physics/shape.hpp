#pragma once
#include "saga/core/properties.hpp"
#include "saga/physics/quantities.hpp"

namespace saga::physics {

  namespace detail {
    struct radius {};
  } // namespace detail

  // Radius
  template <class TypeDescriptor>
  using radius =
      saga::core::property_configuration<detail::radius,
                                         typename TypeDescriptor::float_type,
                                         TypeDescriptor::backend>;

  template <class TypeDescriptor, template <class T> class... Property>
  struct shape {
    using properties = saga::properties<Property...>;
  };

  template <class TypeDescriptor> struct point : shape<TypeDescriptor> {};

  template <class TypeDescriptor>
  struct sphere : shape<TypeDescriptor, radius> {};
} // namespace saga::physics
