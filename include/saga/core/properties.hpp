#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/fields.hpp"

#if SAGA_CUDA_ENABLED
#include "saga/core/cuda/vector.hpp"
#endif

#include <vector>

namespace saga::core {

  /// Container for the given backend
  template <class T, saga::backend Backend> struct container;

  /// Container for the CPU backend
  template <class T> struct container<T, saga::backend::CPU> {
    using type = std::vector<T>;
  };

#if SAGA_CUDA_ENABLED
  /// Container for the CUDA backend
  template <class T> struct container<T, saga::backend::CUDA> {
    using type = saga::core::cuda::vector<T, saga::core::cuda::host>;
  };
#endif

  /// Alias to get the type of a container for a given backend
  template <class T, backend Backend>
  using container_t = typename container<T, Backend>::type;

  /*!\brief
   */
  template <class FieldName, class T, saga::backend Backend>
  struct property_configuration {

    /// Alias for the property template used to access the fields
    template <class TypeDescriptor>
    using property_template = property_configuration<FieldName, T, Backend>;

    using underlying_value_type = T;
    using underlying_container_type =
        container_t<underlying_value_type, Backend>;
  };

  template <class T> struct underlying_value_type {
    using type = typename T::underlying_value_type;
  };

  template <class T>
  using underlying_value_type_t = typename underlying_value_type<T>::type;
} // namespace saga::core
