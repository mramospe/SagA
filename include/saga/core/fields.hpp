#pragma once
#include "saga/core/types.hpp"
#include "saga/core/utils.hpp"
#include <tuple>
#include <type_traits>

namespace saga::core {

  namespace fields {

    /// Field with an associated floating-point value
    template <class TypeDescriptor, class Enable = void>
    struct floating_point_field;

    /// Field with an associated floating-point value
    template <class TypeDescriptor>
    struct floating_point_field<
        TypeDescriptor,
        std::enable_if_t<
            saga::types::is_valid_type_descriptor_v<TypeDescriptor>>> {
      using float_type = typename TypeDescriptor::float_type;
    };

    /// Collection of fields
    template <class... Field> using fields_pack = std::tuple<Field...>;

    /// Expand a set of fields using the given float type and backend as
    /// arguments
    template <class T, template <class> class... Field> struct expand_fields {
      using type = fields_pack<Field<T>...>;
    };

    /// Expand a set of fields using the given float type and backend as
    /// arguments
    template <class T, template <class> class... Fields>
    using expand_fields_t = typename expand_fields<T, Fields...>::type;

    /// Extend fields
    template <class... Fields> struct extend_fields;

    /// Add a new field to a set of fields
    template <class F0, class... Fields>
    struct extend_fields<fields_pack<Fields...>, F0> {
      using type = fields_pack<Fields..., F0>;
    };

    /// Add a new field to a set of fields
    template <class F0, class... Fields>
    struct extend_fields<F0, fields_pack<Fields...>> {
      using type = fields_pack<F0, Fields...>;
    };

    /// Concatenate two sets of fields
    template <class... F0, class... F1>
    struct extend_fields<fields_pack<F0...>, fields_pack<F1...>> {
      using type = fields_pack<F0..., F1...>;
    };

    /// Alias to extend fields
    template <class... Fields>
    using extend_fields_t = typename extend_fields<Fields...>::type;

    /// Helper to apply functions on containers
    template <class Fields> struct visitor;

    /// Helper to apply functions on containers
    template <class... Fields> struct visitor<fields_pack<Fields...>> {

      /// Apply the function only on the first field container and return its
      /// content
      template <class Function, class Container, class... Args>
      static auto invoke_first(Function fn, Container const &cont,
                               Args &&... args) {
        return fn(
            saga::core::type_at_t<0, Fields...>::container_type::get(cont),
            args...);
      }

      /// Apply the function in each of the field containers
      template <class Function, class Container, class... Args>
      static void invoke_void(Function fn, Container const &cont,
                              Args &&... args) {
        (fn(Fields::container_type::get(cont), args...), ...);
      }

      /// Apply the function in each of the field containers
      template <class Function, class Container, class... Args>
      static void invoke_void(Function fn, Container &cont, Args &&... args) {
        (fn(Fields::container_type::get(cont), args...), ...);
      }
    };

  } // namespace fields

} // namespace saga::core
