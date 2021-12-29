#pragma once
#include <cstdlib>

namespace saga::core {

  /// Whether the type is in the given list
  template <class Reference, class... T> struct has_type : std::false_type {};

  /// Whether the type is in the given list
  template <class Reference, class... T>
  struct has_type<Reference, Reference, T...> : std::true_type {};

  /// Whether the type is in the given list
  template <class Reference, class T0, class... T>
  struct has_type<Reference, T0, T...> : has_type<Reference, T...> {};

  /// Whether the type is in the given list
  template <class Reference, class... T>
  static constexpr auto has_type_v = has_type<Reference, T...>::value;

  /// Check if a list of template arguments has repeated types
  template <class... T>
  struct has_repeated_template_arguments : std::false_type {};

  /// Check if a list of template arguments has repeated types
  template <class T0, class... T>
  struct has_repeated_template_arguments<T0, T...>
      : std::conditional_t<has_type_v<T0, T...>, std::true_type,
                           has_repeated_template_arguments<T...>> {};

  /// Check if a list of template arguments has repeated types
  template <class... T>
  static constexpr auto has_repeated_template_arguments_v =
      has_repeated_template_arguments<T...>::value;

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref, template <class> class... T>
  struct has_single_template : std::false_type {};

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref>
  struct has_single_template<Ref> : std::false_type {};

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref, template <class> class T0,
            template <class> class... T>
  struct has_single_template<Ref, T0, T...> : has_single_template<Ref, T...> {};

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref, template <class> class... T>
  struct has_single_template<Ref, Ref, T...> : std::true_type {};

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref, template <class> class... T>
  static constexpr auto has_single_template_v =
      has_single_template<Ref, T...>::value;

  /// Get the index of the type in the list of types
  template <class Match, class... T> struct index;

  /// Get the index of the type in the list of types
  template <class Match, class T0, class... T> struct index<Match, T0, T...> {
    static constexpr auto value = index<Match, T...>::value + 1;
  };

  /// Get the index of the type in the list of types
  template <class Match, class... T> struct index<Match, Match, T...> {
    static_assert(!has_type_v<Match, T...>,
                  "Multiple matches found for the given type");
    static constexpr auto value = 0u;
  };

  /// Get the type at the given position
  template <class Match, class... T>
  static constexpr auto index_v = index<Match, T...>::value;

  /// Get the index of the type in the list of types
  template <template <class> class Match, template <class> class... T>
  struct template_index;

  /// Get the index of the type in the list of types
  template <template <class> class Match, template <class> class T0,
            template <class> class... T>
  struct template_index<Match, T0, T...> {
    static constexpr auto value = template_index<Match, T...>::value + 1;
  };

  /// Get the index of the type in the list of types
  template <template <class> class Match, template <class> class... T>
  struct template_index<Match, Match, T...> {
    static constexpr auto value = 0u;
  };

  /// Get the type at the given position
  template <template <class> class Match, template <class> class... T>
  static constexpr auto template_index_v = template_index<Match, T...>::value;

  /// Get the type at the given position
  template <std::size_t I, class T0, class... T> struct type_at {
    using type = typename type_at<I - 1, T...>::type;
  };

  /// Get the type at the given position
  template <class T0, class... T> struct type_at<0, T0, T...> {
    using type = T0;
  };

  /// Get the template argument at the given position
  template <std::size_t I, class... T>
  using type_at_t = typename type_at<I, T...>::type;

  /// Get the template argument at the given position
  template <std::size_t I, template <class> class T0,
            template <class> class... T>
  struct template_at {
    template <class V>
    using tpl = typename template_at<I - 1, T...>::template tpl<V>;
  };

  /// Get the template argument at the given position
  template <template <class> class T0, template <class> class... T>
  struct template_at<0, T0, T...> {
    template <class V> using tpl = T0<V>;
  };

  /// Get the value at the given position
  template <std::size_t I, class T0, class... T>
  constexpr auto const &value_at(T0 const &v0, T const &...v) {
    static_assert(I < 1 + sizeof...(T), "Template index is out of range");
    if constexpr (I == 0)
      return v0;
    else
      return value_at<I - 1>(v...);
  }

  /// Get the value at the given position
  template <std::size_t I, class T0, class... T>
  constexpr auto &value_at(T0 &v0, T &...v) {
    static_assert(I < 1 + sizeof...(T), "Template index is out of range");
    if constexpr (I == 0)
      return v0;
    else
      return value_at<I - 1>(v...);
  }

} // namespace saga::core
