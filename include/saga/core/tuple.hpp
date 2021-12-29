#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/utils.hpp"
#include <cstdlib>
#include <utility>

namespace saga::core {

  namespace detail {
    // forward declaration
    template <std::size_t I> struct get_t;
  } // namespace detail

  template <class... T> class tuple;

  template <class T, class... Ts> class tuple<T, Ts...> {

    using size_type = std::size_t;

    template <std::size_t> friend struct detail::get_t;

  public:
    tuple() = default;
    tuple(tuple &&) = default;
    tuple(tuple const &) = default;
    __saga_core_function__ tuple(T &&t, Ts &&...r)
        : m_value{std::forward<T>(t)}, m_remainder{std::forward<Ts>(r)...} {}
    __saga_core_function__ tuple(T const &t, Ts const &...r)
        : m_value{t}, m_remainder{r...} {}

    tuple &operator=(tuple const &) = default;
    tuple &operator=(tuple &&) = default;

    __saga_core_function__ constexpr size_type size() const {
      return sizeof...(Ts) + 1;
    }

  protected:
    T m_value;
    tuple<Ts...> m_remainder;
  };

  template <class T> class tuple<T> {

    using size_type = std::size_t;

    template <std::size_t> friend struct detail::get_t;

  public:
    tuple() = default;
    tuple(tuple &&) = default;
    tuple(tuple const &) = default;
    __saga_core_function__ tuple(T &&t) : m_value{std::forward<T>(t)} {}
    __saga_core_function__ tuple(T const &t) : m_value{t} {}

    tuple &operator=(tuple const &) = default;
    tuple &operator=(tuple &&) = default;

    __saga_core_function__ constexpr size_type size() const { return 1u; }

  protected:
    T m_value;
  };

  template <> class tuple<> {

    using size_type = std::size_t;

  public:
    tuple() = default;
    tuple(tuple &&) = default;
    tuple(tuple const &) = default;

    tuple &operator=(tuple const &) = default;
    tuple &operator=(tuple &&) = default;

    __saga_core_function__ constexpr size_type size() const { return 0u; }
  };

  namespace detail {

    //
    // This duplication is needed to avoid getting warnings of the type
    //
    // warning: missing return statement at end of non-void function
    //
    // when using "if constexpr" expressions with nvcc
    //

    template <std::size_t I> struct get_t {

      template <class... T>
      __saga_core_function__ constexpr auto const &
      operator()(tuple<T...> const &t) const {
        return get_t<I - 1>{}(t.m_remainder);
      }

      template <class... T>
      __saga_core_function__ constexpr auto &operator()(tuple<T...> &t) const {
        return get_t<I - 1>{}(t.m_remainder);
      }
    };

    template <> struct get_t<0> {

      template <class... T>
      __saga_core_function__ constexpr auto const &
      operator()(tuple<T...> const &t) const {
        return t.m_value;
      }

      template <class... T>
      __saga_core_function__ constexpr auto &operator()(tuple<T...> &t) const {
        return t.m_value;
      }
    };
  } // namespace detail

  template <class... T> __saga_core_function__ auto make_tuple(T &&...t) {
    return tuple<T...>(std::forward<T>(t)...);
  }

  template <class... T> __saga_core_function__ auto make_tuple(T const &...t) {
    return tuple<T...>(t...);
  }

  template <class T, class... Ts>
  __saga_core_function__ auto &get(tuple<Ts...> &t) {
    return get<saga::core::index_v<T, Ts...>>(t);
  }

  template <class T, class... Ts>
  __saga_core_function__ auto const &get(tuple<Ts...> const &t) {
    return get<saga::core::index_v<T, Ts...>>(t);
  }

  template <std::size_t I, class... Ts>
  __saga_core_function__ auto &get(tuple<Ts...> &t) {
    return detail::get_t<I>{}(t);
  }

  template <std::size_t I, class... Ts>
  __saga_core_function__ auto const &get(tuple<Ts...> const &t) {
    return detail::get_t<I>{}(t);
  }
} // namespace saga::core
