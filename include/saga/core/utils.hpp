#pragma once
#include <tuple>

namespace saga::core {

  template <class Match, class... T> struct index;

  template <class Match, class T0, class... T> struct index<Match, T0, T...> {
    static constexpr auto value = index<Match, T...>::value;
  };

  template <class Match, class... T> struct index<Match, Match, T...> {
    static constexpr auto value = 0u;
  };

  template <class Match> struct index<Match> {};

  template <class Match, class... T>
  static constexpr auto index_v = index<Match, T...>::value;

  template <std::size_t I, class T0, class... T> struct type_at {
    using type = typename type_at<I - 1, T...>::type;
  };

  template <class T0, class... T> struct type_at<0, T0, T...> {
    using type = T0;
  };

  template <std::size_t I, class... T>
  using type_at_t = typename type_at<I, T...>::type;

  template <std::size_t I, template <class> class T0,
            template <class> class... T>
  struct template_at {
    template <class V>
    using tpl = typename template_at<I - 1, T...>::template tpl<V>;
  };

  template <template <class> class T0, template <class> class... T>
  struct template_at<0, T0, T...> {
    template <class V> using tpl = T0<V>;
  };
} // namespace saga::core
