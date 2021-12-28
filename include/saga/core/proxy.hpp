#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/utils.hpp"

#include <tuple>

namespace saga::core {

  namespace detail {

    template <class Container, class Fields> class value;

    template <class Container, class Fields> class proxy;

    template <class Container, class Fields> class const_proxy;

    /* \brief A container value type
       This is not the actual type stored by the container, but rather a proxy
       to do operations with elements of a container.
     */
    template <class Container, template <class> class... Field>
    class value<Container, saga::properties<Field...>>
        : protected std::tuple<saga::core::underlying_value_type_t<
              Field<typename Container::type_descriptor>>...> {

    public:
      using type_descriptor = typename Container::type_descriptor;

      using base_type = std::tuple<
          saga::core::underlying_value_type_t<Field<type_descriptor>>...>;

      using fields_type = saga::properties<Field...>;

      using proxy_type = proxy<Container, fields_type>;
      using const_proxy_type = const_proxy<Container, fields_type>;

      value() = default;
      __saga_core_function__
      value(saga::core::underlying_value_type_t<Field<type_descriptor>> &&...v)
          : base_type(std::forward<saga::core::underlying_value_type_t<
                          Field<type_descriptor>>>(v)...) {}
      __saga_core_function__
      value(saga::core::underlying_value_type_t<Field<type_descriptor>> const
                &...v)
          : base_type(v...) {}

      value(value const &) = default;
      value(value &&) = default;
      value &operator=(value &&) = default;
      value &operator=(value const &) = default;

      __saga_core_function__ value(proxy_type const &p)
          : value(p.template get<Field>()...){};

      __saga_core_function__ value &operator=(proxy_type const &p) {
        (set<Field>(p.template get<Field>()), ...);
        return *this;
      }
      __saga_core_function__ value &operator=(const_proxy_type const &p) {
        (set<Field>(p.template get<Field>()), ...);
        return *this;
      }

      /// Get the value of the given field
      template <template <class> class F>
      __saga_core_function__ auto const &get() const {
        return std::get<saga::core::template_index_v<F, Field...>>(*this);
      }

      /// Get the value of the given field
      template <template <class> class F> __saga_core_function__ auto &get() {
        return std::get<saga::core::template_index_v<F, Field...>>(*this);
      }

      /// Whether this class has the specified property
      template <template <class> class Property>
      constexpr __saga_core_function__ bool has() const {
        return saga::core::is_template_in_v<Property, Field...>;
      }

      /// Set the values of all the fields
      template <template <class> class F>
      __saga_core_function__ void
      set(saga::core::underlying_value_type_t<F<type_descriptor>> v) {
        std::get<saga::core::template_index_v<F, Field...>>(*this) = v;
      }
    };

    /* \brief A container proxy type
       This object is returned by containers when accessing a single element
     */
    template <class Container, template <class> class... Field>
    class proxy<Container, saga::properties<Field...>> {

    public:
      using type_descriptor = typename Container::type_descriptor;

      /// Container type
      using container_type = Container;
      using container_pointer_type = container_type *;
      using size_type = typename Container::size_type;

      using fields_type = saga::properties<Field...>;

      using value_type = value<Container, fields_type>;
      using const_proxy_type = const_proxy<Container, fields_type>;

      /// Build the proxy from the container and the index
      __saga_core_function__ proxy(container_pointer_type cont, size_type idx)
          : m_ptr{cont}, m_idx{idx} {}
      /// The copy constructor assigns the internal container and index from the
      /// argument
      proxy(const proxy &) = default;
      /// The move constructor assigns the internal container and index from the
      /// argument
      proxy(proxy &&) = default;

      __saga_core_function__ proxy &operator=(proxy const &other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }
      __saga_core_function__ proxy &operator=(proxy &&other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }

      __saga_core_function__ proxy &operator=(const_proxy_type const &other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }
      __saga_core_function__ proxy &operator=(const_proxy_type &&other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }

      /// Assignment operator from a value type
      __saga_core_function__ proxy &operator=(value_type const &other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }

      /// Assignment operator from a value type
      __saga_core_function__ proxy &operator=(value_type &&other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }

      /// Set each element in the associated field of the container
      template <template <class> class F>
      __saga_core_function__ void
      set(saga::core::underlying_value_type_t<F<type_descriptor>> v) {

        m_ptr->template set<F>(m_idx, v);
      }

      /// Whether this class has the specified property
      template <template <class> class Property>
      constexpr __saga_core_function__ bool has() const {
        return saga::core::is_template_in_v<Property, Field...>;
      }
      /// Get the value of one field from the container
      template <template <class> class F>
      __saga_core_function__ auto const &get() const {
        return m_ptr->template get<F>()[m_idx];
      }
      /// Get the value of one field from the container
      template <template <class> class F> __saga_core_function__ auto &get() {
        return m_ptr->template get<F>()[m_idx];
      }

    protected:
      /// Pointer to the container
      __saga_core_function__ container_pointer_type container_ptr() {
        return m_ptr;
      }
      /// Container as a reference
      __saga_core_function__ container_type &container() { return *m_ptr; }
      /// Container as a reference
      __saga_core_function__ container_type const &container() const {
        return *m_ptr;
      }
      /// Current index this proxy points to
      __saga_core_function__ constexpr size_type index() const { return m_idx; }

      /// Pointer to the container
      container_pointer_type m_ptr = nullptr;
      /// Index in the container
      size_type m_idx = 0;
    };

    /* \brief A container proxy type
       This object is returned by containers when accessing a single element
     */
    template <class Container, template <class> class... Field>
    class const_proxy<Container, saga::properties<Field...>> {

    public:
      using type_descriptor = typename Container::type_descriptor;

      /// Container type
      using container_type = Container;
      using container_pointer_type = container_type const *;
      using size_type = typename Container::size_type;

      using fields_type = saga::properties<Field...>;

      using proxy_type = proxy<Container, fields_type>;

      /// Build the proxy from the container and the index
      __saga_core_function__ const_proxy(container_pointer_type cont,
                                         size_type idx)
          : m_ptr{cont}, m_idx{idx} {}
      /// The copy constructor assigns the internal container and index from the
      /// argument
      const_proxy(const const_proxy &) = default;
      /// The move constructor assigns the internal container and index from the
      /// argument
      const_proxy(const_proxy &&) = default;

      /// Whether this class has the specified property
      template <template <class> class Property>
      constexpr __saga_core_function__ bool has() const {
        return saga::core::is_template_in_v<Property, Field...>;
      }
      /// Get the value of the field
      template <template <class> class F>
      __saga_core_function__ auto const &get() const {
        return m_ptr->template get<F>()[m_idx];
      }

    protected:
      /// Pointer to the container
      __saga_core_function__ container_pointer_type container_ptr() const {
        return m_ptr;
      }
      /// Container as a reference
      __saga_core_function__ container_type const &container() const {
        return *m_ptr;
      }
      /// Current index this proxy points to
      __saga_core_function__ constexpr size_type index() const { return m_idx; }

      /// Pointer to the container
      container_pointer_type m_ptr = nullptr;
      /// Index in the container
      size_type m_idx = 0;
    };
  } // namespace detail

  template <class Container>
  using value = detail::value<Container, typename Container::fields_type>;

  template <class Container>
  using proxy = detail::proxy<Container, typename Container::fields_type>;

  template <class Container>
  using const_proxy =
      detail::const_proxy<Container, typename Container::fields_type>;
} // namespace saga::core
