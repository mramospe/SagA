#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/properties.hpp"
#include <limits>
#include <type_traits>

namespace saga {

  namespace cpu {

    /// Floating-point number with single precision
    struct single_float_precision {
      static constexpr auto backend = saga::backend::CPU;
      using float_type = float;
      using int_type = int;
    };

    /// Alias for \ref single_float_precision
    using sf = single_float_precision;

    /// Floating-point number with double precision
    struct double_float_precision {
      static constexpr auto backend = saga::backend::CPU;
      using float_type = double;
      using int_type = int;
    };

    /// Alias for \ref double_float_precision
    using df = double_float_precision;
  } // namespace cpu

  template <class TypeDescriptor> struct numeric_info;

  /// Numerical information for \ref saga::cpu::single_float_precision
  template <> struct numeric_info<cpu::single_float_precision> {
    static constexpr auto min = std::numeric_limits<
        typename cpu::single_float_precision::float_type>::min();
    static constexpr auto lowest = std::numeric_limits<
        typename cpu::single_float_precision::float_type>::lowest();
    static constexpr auto max = std::numeric_limits<
        typename cpu::single_float_precision::float_type>::max();
  };

  /// Numerical information for \ref saga::cpu::double_float_precision
  template <> struct numeric_info<cpu::double_float_precision> {
    static constexpr auto min = std::numeric_limits<
        typename cpu::double_float_precision::float_type>::min();
    static constexpr auto lowest = std::numeric_limits<
        typename cpu::double_float_precision::float_type>::lowest();
    static constexpr auto max = std::numeric_limits<
        typename cpu::double_float_precision::float_type>::max();
  };

  namespace core {

    /// Check if the input type is a valid type descriptor
    template <class T> struct is_valid_type_descriptor : std::false_type {};

    /// Check if the input type is a valid type descriptor
    template <>
    struct is_valid_type_descriptor<cpu::single_float_precision>
        : std::true_type {};

    /// Check if the input type is a valid type descriptor
    template <>
    struct is_valid_type_descriptor<cpu::double_float_precision>
        : std::true_type {};

    /// Check if the input type is a valid type descriptor
    template <class T>
    static constexpr auto is_valid_type_descriptor_v =
        is_valid_type_descriptor<T>::value;

  } // namespace core
} // namespace saga
