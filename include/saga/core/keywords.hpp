#pragma once
#include "saga/core/utils.hpp"
#include <tuple>

namespace saga::core::keywords {

  template <template <class> class... Keyword> struct required {};

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

  public:
    using base_type =
        std::tuple<typename RequiredKeyword<TypeDescriptor>::value_type...,
                   typename Keyword<TypeDescriptor>::value_type...>;
    using tuple_type = base_type;

    template <class... Default, class... K>
    keywords_parser(std::tuple<Default...> &&defaults, K... v)
        : base_type{parse_keywords_with_defaults_and_required(
              std::forward<std::tuple<Default...>>(defaults), v...)} {}

    template <template <class> class K> auto get() const {
      return std::get<saga::core::index_v<K<TypeDescriptor>,
                                          RequiredKeyword<TypeDescriptor>...,
                                          Keyword<TypeDescriptor>...>>(*this);
    }

    template <template <class> class K>
    auto set(typename K<TypeDescriptor>::value_type v) const {
      std::get<saga::core::index_v<K, RequiredKeyword<TypeDescriptor>...,
                                   Keyword<TypeDescriptor>...>>(*this) = v;
    }

  private:
    template <std::size_t I, class... Default, class... K>
    static auto value_or_default(std::tuple<Default...> const &defaults,
                                 K... keyword) {

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

    template <std::size_t... I, class... Default, class... K>
    static tuple_type
    parse_keywords_with_defaults_impl(std::index_sequence<I...>,
                                      std::tuple<Default...> &&defaults,
                                      K... keyword) {

      return {value_or_default<I>(defaults, keyword...)...};
    }

    template <class... Default, class... K>
    static tuple_type
    parse_keywords_with_defaults_and_required(std::tuple<Default...> &&defaults,
                                              K... keywords) {
      static_assert(!has_repeated_template_arguments_v<K...>);
      static_assert(!(
          saga::core::has_type_v<RequiredKeyword<TypeDescriptor>, Default...> &&
          ...));
      static_assert(
          (saga::core::has_type_v<RequiredKeyword<TypeDescriptor>, K...> &&
           ...));
      return parse_keywords_with_defaults_impl(
          std::make_index_sequence<((sizeof...(RequiredKeyword)) +
                                    (sizeof...(Keyword)))>(),
          std::forward<std::tuple<Default...>>(defaults), keywords...);
    }
  };
} // namespace saga::core::keywords
