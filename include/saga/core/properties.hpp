#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/fields.hpp"

#include <vector>

namespace saga::core {

  /// Container for the given backend
  template <class T, saga::backend Backend> struct container;

  /// Container for the CPU backend
  template <class T> struct container<T, saga::backend::CPU> {
    using type = std::vector<T>;
  };

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

    /*!\brief
     */
    class value_type {

    public:
      using underlying_value_type = T;

      value_type() = default;
      value_type(underlying_value_type v) : m_value{v} {};
      value_type(value_type const &) = default;
      value_type(value_type &&) = default;
      value_type &operator=(value_type const &) = default;
      value_type &operator=(value_type &&) = default;

      static underlying_value_type const &get(value_type const &obj) {
        return obj.m_value;
      }
      static void set(value_type &obj, underlying_value_type v) {
        obj.m_value = v;
      }

    private:
      underlying_value_type m_value;
    };

    /*!\brief
     */
    class container_type {
    public:
      using underlying_value_type = T;
      using underlying_container_type =
          container_t<underlying_value_type, Backend>;
      container_type() = default;
      container_type(container_type const &other) = default;
      container_type(container_type &&other) = default;
      container_type &operator=(container_type const &other) = default;
      container_type &operator=(container_type &&other) = default;
      container_type(std::size_t n) : m_container(n) {}

      static underlying_container_type const &get(container_type const &cont) {
        return cont.m_container;
      }
      static underlying_container_type &get(container_type &cont) {
        return cont.m_container;
      }
      static underlying_value_type const &get(container_type const &cont,
                                              std::size_t idx) {
        return cont.m_container[idx];
      }
      static underlying_value_type &get(container_type &cont, std::size_t idx) {
        return cont.m_container[idx];
      }
      static void set(container_type &cont, std::size_t idx,
                      underlying_value_type v) {
        cont.m_container[idx] = v;
      }

    private:
      underlying_container_type m_container;
    };

    /*!\brief
     */
    template <class Container> class proxy_type {
    public:
      virtual Container &container() const = 0;
      virtual std::size_t index() const = 0;
      static auto const &get(proxy_type const &obj) {
        return obj.container().template get<property_template>().at(
            obj.index());
      }
      static auto &get(proxy_type &obj) {
        return obj.container().template get<property_template>().at(
            obj.index());
      }
      static void set(proxy_type &obj, underlying_value_type v) {
        obj.container().template set<property_template>(obj.index(), v);
      }
    };

    /*!\brief
     */
    template <class Container> class const_proxy_type {
    public:
      virtual Container const &container() const = 0;
      virtual std::size_t index() const = 0;
      static auto const &get(const_proxy_type const &obj) {
        return obj.container().template get<property_template>().at(
            obj.index());
      }
    };
  };
} // namespace saga::core
