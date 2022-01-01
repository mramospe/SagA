#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/properties.hpp"
#include <limits>
#include <type_traits>

namespace saga {

  namespace core {

    /// Information of a backend and precision to use
    template <backend Backend, class FloatType, class IntType>
    struct type_descriptor {

      static_assert(std::is_floating_point_v<FloatType>,
                    "Must provide a valid floating-point type");
      static_assert(std::is_integral_v<IntType>,
                    "Must provide a valid integral type");

      static constexpr auto backend = Backend;
      using float_type = FloatType;
      using int_type = IntType;
    };
  } // namespace core

  namespace cpu {
    /// Floating-point number with single precision
    using single_float_precision =
        saga::core::type_descriptor<saga::backend::CPU, float, int>;
    /// Alias for \ref single_float_precision
    using sf = single_float_precision;
    /// Floating-point number with double precision
    using double_float_precision =
        saga::core::type_descriptor<saga::backend::CPU, double, int>;
    /// Alias for \ref double_float_precision
    using df = double_float_precision;
  } // namespace cpu

  namespace cuda {
    /// Floating-point number with single precision
    using single_float_precision =
        saga::core::type_descriptor<saga::backend::CUDA, float, int>;
    /// Alias for \ref single_float_precision
    using sf = single_float_precision;
    /// Floating-point number with double precision
    using double_float_precision =
        saga::core::type_descriptor<saga::backend::CUDA, double, int>;
    /// Alias for \ref double_float_precision
    using df = double_float_precision;
  } // namespace cuda

  /// Numerical information for a type descriptor
  template <class TypeDescriptor> struct numeric_info;

  template <backend Backend, class FloatType, class IntType>
  struct numeric_info<
      saga::core::type_descriptor<Backend, FloatType, IntType>> {
    static constexpr auto min = std::numeric_limits<FloatType>::min();
    static constexpr auto lowest = std::numeric_limits<FloatType>::lowest();
    static constexpr auto max = std::numeric_limits<FloatType>::max();
  };

  namespace core {

    /// Represent the given type descriptor but with a different backend
    template <saga::backend NewBackend, class TypeDescriptor>
    struct change_type_descriptor_backend;

    template <backend NewBackend, backend Backend, class FloatType,
              class IntType>
    struct change_type_descriptor_backend<
        NewBackend, saga::core::type_descriptor<Backend, FloatType, IntType>> {

      using type = saga::core::type_descriptor<NewBackend, FloatType, IntType>;
    };

    template <saga::backend NewBackend, class TypeDescriptor>
    using change_type_descriptor_backend_t =
        typename change_type_descriptor_backend<NewBackend,
                                                TypeDescriptor>::type;

    template <class TypeDescriptor> struct switch_type_descriptor_backend;

    template <class FloatType, class IntType>
    struct switch_type_descriptor_backend<
        saga::core::type_descriptor<saga::backend::CPU, FloatType, IntType>> {
      using type =
          saga::core::type_descriptor<saga::backend::CUDA, FloatType, IntType>;
    };

    template <class FloatType, class IntType>
    struct switch_type_descriptor_backend<
        saga::core::type_descriptor<saga::backend::CUDA, FloatType, IntType>> {
      using type =
          saga::core::type_descriptor<saga::backend::CPU, FloatType, IntType>;
    };

    template <class TypeDescriptor>
    using switch_type_descriptor_backend_t =
        typename switch_type_descriptor_backend<TypeDescriptor>::type;

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
    template <>
    struct is_valid_type_descriptor<cuda::single_float_precision>
        : std::true_type {};

    /// Check if the input type is a valid type descriptor
    template <>
    struct is_valid_type_descriptor<cuda::double_float_precision>
        : std::true_type {};

    /// Check if the input type is a valid type descriptor
    template <class T>
    static constexpr auto is_valid_type_descriptor_v =
        is_valid_type_descriptor<T>::value;

  } // namespace core
} // namespace saga
