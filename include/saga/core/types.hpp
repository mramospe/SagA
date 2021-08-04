#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/properties.hpp"
#include <limits>
#include <type_traits>

namespace saga::types {

  namespace cpu {

    struct single_float_precision {
      static constexpr auto backend = saga::backend::CPU;
      using float_type = float;
      using int_type = int;
    };

    struct double_float_precision {
      static constexpr auto backend = saga::backend::CPU;
      using float_type = double;
      using int_type = int;
    };
  } // namespace cpu

  template <class TypeDescriptor> struct numeric_info;

  template <> struct numeric_info<cpu::single_float_precision> {
    static constexpr auto min = std::numeric_limits<
        typename cpu::single_float_precision::float_type>::min();
    static constexpr auto lowest = std::numeric_limits<
        typename cpu::single_float_precision::float_type>::lowest();
    static constexpr auto max = std::numeric_limits<
        typename cpu::single_float_precision::float_type>::max();
  };

  template <> struct numeric_info<cpu::double_float_precision> {
    static constexpr auto min = std::numeric_limits<
        typename cpu::double_float_precision::float_type>::min();
    static constexpr auto lowest = std::numeric_limits<
        typename cpu::double_float_precision::float_type>::lowest();
    static constexpr auto max = std::numeric_limits<
        typename cpu::double_float_precision::float_type>::max();
  };

  template <class T> struct is_valid_type_descriptor : std::false_type {};

  template <>
  struct is_valid_type_descriptor<cpu::single_float_precision>
      : std::true_type {};

  template <>
  struct is_valid_type_descriptor<cpu::double_float_precision>
      : std::true_type {};

  template <class T>
  static constexpr auto is_valid_type_descriptor_v =
      is_valid_type_descriptor<T>::value;

  template <class TypeDescriptor, class T>
  struct is_type_descriptor_type
      : std::conditional_t<
            (std::is_same_v<T, typename TypeDescriptor::float_type> ||
             std::is_same_v<T, typename TypeDescriptor::int_type>),
            std::true_type, std::false_type> {};

  template <class TypeDescriptor, class T>
  static constexpr auto is_type_descriptor_type_v =
      is_type_descriptor_type<TypeDescriptor, T>::value;

} // namespace saga::types
