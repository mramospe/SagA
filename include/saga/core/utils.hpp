#pragma once
#include <cstdlib>

namespace saga::core {

  /// Get the index of the type in the list of types
  template <class Match, class... T> struct index;

  /// Get the index of the type in the list of types
  template <class Match, class T0, class... T> struct index<Match, T0, T...> {
    static constexpr auto value = index<Match, T...>::value;
  };

  /// Get the index of the type in the list of types
  template <class Match, class... T> struct index<Match, Match, T...> {
    static constexpr auto value = 0u;
  };

  /// Get the type at the given position
  template <class Match, class... T>
  static constexpr auto index_v = index<Match, T...>::value;

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

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref, template <class> class... T>
  struct is_template_in : std::false_type {};

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref>
  struct is_template_in<Ref> : std::false_type {};

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref, template <class> class T0,
            template <class> class... T>
  struct is_template_in<Ref, T0, T...> : is_template_in<Ref, T...> {};

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref, template <class> class... T>
  struct is_template_in<Ref, Ref, T...> : std::true_type {};

  /// Check at compile-time if a reference template is in the list of template
  /// arguments
  template <template <class> class Ref, template <class> class... T>
  static constexpr auto is_template_in_v = is_template_in<Ref, T...>::value;
} // namespace saga::core
