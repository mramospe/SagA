#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/tuple.hpp"
#include "saga/core/utils.hpp"
#include <variant>

namespace saga::core::keywords {

  namespace detail {

    /// Expand a std::variant object with a new type, if it does not contain it
    /// yet
    template <class Variant, class NewType, class Enable = void>
    struct expand_variant;

    /// Expand a std::variant object with a new type, if it does not contain it
    /// yet
    template <class... T, class NewType>
    struct expand_variant<
        std::variant<T...>, NewType,
        std::enable_if_t<!saga::core::has_type_v<NewType, T...>>> {
      using type = std::variant<T..., NewType>;
    };

    /// Expand a std::variant object with a new type, if it does not contain it
    /// yet
    template <class... T, class NewType>
    struct expand_variant<
        std::variant<T...>, NewType,
        std::enable_if_t<saga::core::has_type_v<NewType, T...>>> {
      using type = std::variant<T...>;
    };

    /// Expand a std::variant object with a new type, if it does not contain it
    /// yet
    template <class Variant, class NewType>
    using expand_variant_t = typename expand_variant<Variant, NewType>::type;

    /// Expand a std::variant with several types, avoiding repetitions
    template <class Variant, class... T> struct expand_variant_with_types;

    /// Expand a std::variant with several types, avoiding repetitions
    template <class... T> struct expand_variant_with_types<std::variant<T...>> {
      using type = std::variant<T...>;
    };

    /// Expand a std::variant with several types, avoiding repetitions
    template <class... VariantTypes, class NewType, class... T>
    struct expand_variant_with_types<std::variant<VariantTypes...>, NewType,
                                     T...> {
      using type = typename expand_variant_with_types<
          expand_variant_t<std::variant<VariantTypes...>, NewType>, T...>::type;
    };

    /// Expand a std::variant with several types, avoiding repetitions
    template <class Variant, class... NewTypes>
    using expand_variant_with_types_t =
        typename expand_variant_with_types<Variant, NewTypes...>::type;

    /// Expand a std::variant with several floating-point types, avoiding
    /// repetitions, from type descriptors
    template <class Variant, class... TypeDescriptors>
    using expand_variant_from_type_descriptors_float_t =
        expand_variant_with_types_t<Variant,
                                    typename TypeDescriptors::float_type...>;

    /// Expand a std::variant with several integral types, avoiding repetitions,
    /// from type descriptors
    template <class Variant, class... TypeDescriptors>
    using expand_variant_from_type_descriptors_int_t =
        expand_variant_with_types_t<Variant,
                                    typename TypeDescriptors::int_type...>;
  } // namespace detail

  /// Basic keyword for floating-point numbers
  struct keyword_float {
    using underlying_keyword_type = keyword_float;
    // it seems we can not inherit from the variant type and expect std::visit
    // to work correctly in GCC 9.3.0
    detail::expand_variant_from_type_descriptors_float_t<
        std::variant<>, saga::cpu::sf, saga::cpu::df, saga::cuda::sf,
        saga::cuda::df>
        variant;
  };

  /// Basic keyword for integral numbers
  struct keyword_int {
    using underlying_keyword_type = keyword_int;
    // it seems we can not inherit from the variant type and expect std::visit
    // to work correctly in GCC 9.3.0
    detail::expand_variant_from_type_descriptors_int_t<
        std::variant<>, saga::cpu::sf, saga::cpu::df, saga::cuda::sf,
        saga::cuda::df>
        variant;
  };

  namespace detail {

    /// Determine the numeric type of a keyword for a type descriptor
    template <class TypeDescriptor, class KeywordType>
    struct keyword_numeric_type;

    /// Determine the numeric type of a keyword for a type descriptor
    template <class TypeDescriptor>
    struct keyword_numeric_type<TypeDescriptor, keyword_float> {
      using type = typename TypeDescriptor::float_type;
    };

    /// Determine the numeric type of a keyword for a type descriptor
    template <class TypeDescriptor>
    struct keyword_numeric_type<TypeDescriptor, keyword_int> {
      using type = typename TypeDescriptor::int_type;
    };

    /// Determine the numeric type of a keyword for a type descriptor
    template <class TypeDescriptor, class KeywordType>
    using keyword_numeric_type_t =
        typename keyword_numeric_type<TypeDescriptor, KeywordType>::type;
  } // namespace detail

  /// Set of required keyword arguments
  template <class... Keyword> struct required {};

  /*!\brief Class that accepts keyword arguments in the constructor

    Keyword arguments are wrappers around floating point and integral values
    that are used in order to avoid having several inputs values with no visible
    correspondence to parameters in the class. The use of keywords also allows
    to provide the input arguments in any order.

    The keyword arguments are stored within the class, which inherits from
    \ref saga::core::tuple. You can use the \ref keywords_parser::get and
    \ref keywords_parser::set member functions to manipulate the values.
   */
  template <class TypeDescriptor, class Required, class... Keyword>
  class keywords_parser;

  template <class TypeDescriptor, class... RequiredKeyword, class... Keyword>
  class keywords_parser<TypeDescriptor, required<RequiredKeyword...>,
                        Keyword...>
      : protected saga::core::tuple<
            detail::keyword_numeric_type_t<
                TypeDescriptor,
                typename RequiredKeyword::underlying_keyword_type>...,
            detail::keyword_numeric_type_t<
                TypeDescriptor, typename Keyword::underlying_keyword_type>...> {

    static_assert(
        !has_repeated_template_arguments_v<RequiredKeyword..., Keyword...>,
        "Keyword arguments are repeated");

  public:
    /// Base type
    using base_type = saga::core::tuple<
        detail::keyword_numeric_type_t<
            TypeDescriptor,
            typename RequiredKeyword::underlying_keyword_type>...,
        detail::keyword_numeric_type_t<
            TypeDescriptor, typename Keyword::underlying_keyword_type>...>;

    keywords_parser() = default;
    keywords_parser(keywords_parser const &) = default;
    keywords_parser(keywords_parser &&) = default;
    keywords_parser &operator=(keywords_parser const &) = default;
    keywords_parser &operator=(keywords_parser &&) = default;

    /// Constructor from the keyword arguments and a tuple of default values
    template <class... Default, class... K>
    keywords_parser(saga::core::tuple<Default...> &&defaults, K... v) noexcept
        : base_type{parse_keywords_with_defaults_and_required(
              std::forward<saga::core::tuple<Default...>>(defaults), v...)} {}

    /// Get a keyword argument
    template <class K> __saga_core_function__ constexpr auto get() const {
      return saga::core::get<
          saga::core::index_v<K, RequiredKeyword..., Keyword...>>(*this);
    }

    /// Set a keyword argument
    template <class K>
    __saga_core_function__ constexpr auto set(typename K::value_type v) const {
      saga::core::get<saga::core::index_v<K, RequiredKeyword..., Keyword...>>(
          *this) = v;
    }

  private:
    /*!\brief Parse a single keyword argument

      If a value is not provided, it is taken from the tuple of default values.
    */
    template <std::size_t I, class... Default, class... K>
    static constexpr auto
    value_or_default(saga::core::tuple<Default...> const &defaults,
                     K... keyword) {

      using current_keyword_type =
          saga::core::type_at_t<I, RequiredKeyword..., Keyword...>;

      detail::keyword_numeric_type_t<
          TypeDescriptor,
          typename current_keyword_type::underlying_keyword_type>
          value;

      // conversion to the type from the current type descriptor
      if constexpr (saga::core::has_type_v<current_keyword_type, K...>) {
        std::visit(
            [&](auto const &v) { value = v; },
            saga::core::value_at<
                saga::core::index_v<current_keyword_type, K...>>(keyword...)
                .variant);
      } else {
        std::visit([&](auto const &v) { value = v; },
                   saga::core::get<current_keyword_type>(defaults).variant);
      }

      return value;
    }

    /// Parse the input keyword arguments with the given list of default values
    template <std::size_t... I, class... Default, class... K>
    static constexpr base_type
    parse_keywords_with_defaults_impl(std::index_sequence<I...>,
                                      saga::core::tuple<Default...> &&defaults,
                                      K... keyword) {

      return {value_or_default<I>(defaults, keyword...)...};
    }

    /// Parse the input keyword arguments with the given list of default values
    template <class... Default, class... K>
    static constexpr base_type parse_keywords_with_defaults_and_required(
        saga::core::tuple<Default...> &&defaults, K... keywords) {
      static_assert(!has_repeated_template_arguments_v<K...>,
                    "Keyword arguments are repeated");
      static_assert(
          !(saga::core::has_type_v<RequiredKeyword, Default...> && ...),
          "Required keyword arguments are found in the list of default values");
      static_assert((saga::core::has_type_v<RequiredKeyword, K...> && ...),
                    "Required keyword arguments are not provided");
      return parse_keywords_with_defaults_impl(
          std::make_index_sequence<((sizeof...(RequiredKeyword)) +
                                    (sizeof...(Keyword)))>(),
          std::forward<saga::core::tuple<Default...>>(defaults), keywords...);
    }
  };
} // namespace saga::core::keywords
