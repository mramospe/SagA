#pragma once
#include "saga/core/fields.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/utils.hpp"

namespace saga::property {

  namespace detail {
    struct electric_charge {};
  } // namespace detail

  // Additional
  template <class TypeDescriptor>
  using electric_charge =
      saga::core::property_configuration<detail::electric_charge,
                                         typename TypeDescriptor::float_type,
                                         TypeDescriptor::backend>;

} // namespace saga::property
