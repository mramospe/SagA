#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/fields.hpp"

#if SAGA_CUDA_ENABLED
#include "saga/core/cuda/vector.hpp"
#endif

#include <vector>

namespace saga {
  template <template <class> class... Property> struct properties {};
} // namespace saga

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
    using type = saga::core::cuda::vector<T>;
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

  template <class Properties, template <class> class... P>
  struct append_properties;

  template <template <class> class... P0, template <class> class... P1>
  struct append_properties<saga::properties<P0...>, P1...> {
    using type = properties<P0..., P1...>;
  };

  template <class Properties, template <class> class... P>
  using append_properties_t =
      typename append_properties<Properties, P...>::type;
} // namespace saga::core

/// Properties defined as struct types
namespace saga::property {

  namespace detail {
    struct x {};
    struct y {};
    struct z {};
    struct t {};
    struct px {};
    struct py {};
    struct pz {};
    struct e {};
  } // namespace detail

  // Position
  template <class TypeDescriptor>
  using x = saga::core::property_configuration<
      detail::x, typename TypeDescriptor::float_type, TypeDescriptor::backend>;
  template <class TypeDescriptor>
  using y = saga::core::property_configuration<
      detail::y, typename TypeDescriptor::float_type, TypeDescriptor::backend>;
  template <class TypeDescriptor>
  using z = saga::core::property_configuration<
      detail::z, typename TypeDescriptor::float_type, TypeDescriptor::backend>;
  template <class TypeDescriptor>
  using t = saga::core::property_configuration<
      detail::t, typename TypeDescriptor::float_type, TypeDescriptor::backend>;
  // Momenta
  template <class TypeDescriptor>
  using px = saga::core::property_configuration<
      detail::px, typename TypeDescriptor::float_type, TypeDescriptor::backend>;
  template <class TypeDescriptor>
  using py = saga::core::property_configuration<
      detail::py, typename TypeDescriptor::float_type, TypeDescriptor::backend>;
  template <class TypeDescriptor>
  using pz = saga::core::property_configuration<
      detail::pz, typename TypeDescriptor::float_type, TypeDescriptor::backend>;
  template <class TypeDescriptor>
  using e = saga::core::property_configuration<
      detail::e, typename TypeDescriptor::float_type, TypeDescriptor::backend>;
} // namespace saga::property
