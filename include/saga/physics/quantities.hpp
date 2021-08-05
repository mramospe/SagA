#pragma once
#include "saga/core/fields.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/utils.hpp"

namespace saga {

  template <template <class T> class... Property> struct properties {};

  namespace property {

    namespace detail {
      struct x {};
      struct y {};
      struct z {};
      struct t {};
      struct px {};
      struct py {};
      struct pz {};
      struct e {};
      struct electric_charge {};
    } // namespace detail

    // Position
    template <class TypeDescriptor>
    using x =
        saga::core::property_configuration<detail::x,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;
    template <class TypeDescriptor>
    using y =
        saga::core::property_configuration<detail::y,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;
    template <class TypeDescriptor>
    using z =
        saga::core::property_configuration<detail::z,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;
    template <class TypeDescriptor>
    using t =
        saga::core::property_configuration<detail::t,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;
    // Momenta
    template <class TypeDescriptor>
    using px =
        saga::core::property_configuration<detail::px,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;
    template <class TypeDescriptor>
    using py =
        saga::core::property_configuration<detail::py,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;
    template <class TypeDescriptor>
    using pz =
        saga::core::property_configuration<detail::pz,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;
    template <class TypeDescriptor>
    using e =
        saga::core::property_configuration<detail::e,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;

    // Additional
    template <class TypeDescriptor>
    using electric_charge =
        saga::core::property_configuration<detail::electric_charge,
                                           typename TypeDescriptor::float_type,
                                           TypeDescriptor::backend>;
  } // namespace property
} // namespace saga
