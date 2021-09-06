#pragma once
#include "saga/core/utils.hpp"
#include <tuple>

namespace saga::core::keywords {

  /// Set of required keyword arguments
  template <template <class> class... Keyword> struct required {};

  /*!\brief Class that accepts keyword arguments in the constructor

    Keyword arguments are wrappers around floating point and integral values
    that are used in order to avoid having several inputs values with no visible
    correspondence to parameters in the class. The use of keywords also allows
    to provide the input arguments in any order.

    The keyword arguments are stored within the class, which inherits from
    \ref std::tuple. You can use the \ref keywords_parser::get and
    \ref keywords_parser::set member functions to manipulate the values.
   */
  template <class TypeDescriptor, class Required,
            template <class> class... Keyword>
  class keywords_parser;

  template <class TypeDescriptor, template <class> class... RequiredKeyword,
            template <class> class... Keyword>
  class keywords_parser<TypeDescriptor, required<RequiredKeyword...>,
                        Keyword...>
      : protected std::tuple<
            typename RequiredKeyword<TypeDescriptor>::value_type...,
            typename Keyword<TypeDescriptor>::value_type...> {

    static_assert(
        !has_repeated_template_arguments_v<RequiredKeyword<TypeDescriptor>...,
                                           Keyword<TypeDescriptor>...>,
        "Keyword arguments are repeated");

  public:
    /// Base type
    using base_type =
        std::tuple<typename RequiredKeyword<TypeDescriptor>::value_type...,
                   typename Keyword<TypeDescriptor>::value_type...>;

    /// Constructor from the keyword arguments and a tuple of default values
    template <class... Default, class... K>
    keywords_parser(std::tuple<Default...> &&defaults, K... v) noexcept
        : base_type{parse_keywords_with_defaults_and_required(
              std::forward<std::tuple<Default...>>(defaults), v...)} {}

    /// Get a keyword argument
    template <template <class> class K> constexpr auto get() const {
      return std::get<saga::core::index_v<K<TypeDescriptor>,
                                          RequiredKeyword<TypeDescriptor>...,
                                          Keyword<TypeDescriptor>...>>(*this);
    }

    /// Set a keyword argument
    template <template <class> class K>
    constexpr auto set(typename K<TypeDescriptor>::value_type v) const {
      std::get<saga::core::index_v<K, RequiredKeyword<TypeDescriptor>...,
                                   Keyword<TypeDescriptor>...>>(*this) = v;
    }

  private:
    /*!\brief Parse a single keyword argument

      If a value is not provided, it is taken from the tuple of default values.
    */
    template <std::size_t I, class... Default, class... K>
    static constexpr auto
    value_or_default(std::tuple<Default...> const &defaults, K... keyword) {

      using current_keyword_type =
          saga::core::type_at_t<I, RequiredKeyword<TypeDescriptor>...,
                                Keyword<TypeDescriptor>...>;

      if constexpr (saga::core::has_type_v<current_keyword_type, K...>)
        return saga::core::value_at<
                   saga::core::index_v<current_keyword_type, K...>>(keyword...)
            .value;
      else
        return std::get<current_keyword_type>(defaults).value;
    }

    /// Parse the input keyword arguments with the given list of default values
    template <std::size_t... I, class... Default, class... K>
    static constexpr base_type
    parse_keywords_with_defaults_impl(std::index_sequence<I...>,
                                      std::tuple<Default...> &&defaults,
                                      K... keyword) {

      return {value_or_default<I>(defaults, keyword...)...};
    }

    /// Parse the input keyword arguments with the given list of default values
    template <class... Default, class... K>
    static constexpr base_type
    parse_keywords_with_defaults_and_required(std::tuple<Default...> &&defaults,
                                              K... keywords) {
      static_assert(!has_repeated_template_arguments_v<K...>,
                    "Keyword arguments are repeated");
      static_assert(
          !(saga::core::has_type_v<RequiredKeyword<TypeDescriptor>,
                                   Default...> &&
            ...),
          "Required keyword arguments are found in the list of default values");
      static_assert(
          (saga::core::has_type_v<RequiredKeyword<TypeDescriptor>, K...> &&
           ...),
          "Required keyword arguments are not provided");
      return parse_keywords_with_defaults_impl(
          std::make_index_sequence<((sizeof...(RequiredKeyword)) +
                                    (sizeof...(Keyword)))>(),
          std::forward<std::tuple<Default...>>(defaults), keywords...);
    }
  };
} // namespace saga::core::keywords
