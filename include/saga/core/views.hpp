#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/tuple.hpp"

namespace saga::core {

  /// View of a vector that can or can not be modified
  template <class T, saga::backend Backend> class vector_view {

  public:
    using value_type = T;
    using pointer_type = value_type *;
    using reference_type = value_type &;
    using const_reference_type = value_type const &;
    using size_type = std::size_t;

    __saga_core_function__ vector_view(pointer_type ptr, size_type size)
        : m_data{ptr}, m_size{size} {}

    vector_view() = delete;
    vector_view(vector_view const &) = default;
    vector_view(vector_view &&) = default;
    vector_view &operator=(vector_view const &) = default;
    vector_view &operator=(vector_view &&) = default;

    __saga_core_function__ const_reference_type operator[](size_type i) const {
      return m_data[i];
    }

    __saga_core_function__ reference_type operator[](size_type i) {
      return m_data[i];
    }

    __saga_core_function__ constexpr auto size() const { return m_size; }

  private:
    pointer_type m_data;
    size_type m_size;
  };

  /// View of a vector that can not be modified
  template <class T, saga::backend Backend> class const_vector_view {

  public:
    using value_type = T const;
    using pointer_type = value_type *;
    using const_reference_type = value_type const &;
    using size_type = std::size_t;

    __saga_core_function__ const_vector_view(pointer_type ptr, size_type size)
        : m_data{ptr}, m_size{size} {}

    const_vector_view() = delete;
    const_vector_view(const_vector_view const &) = default;
    const_vector_view(const_vector_view &&) = default;
    const_vector_view &operator=(const_vector_view const &) = default;
    const_vector_view &operator=(const_vector_view &&) = default;

    __saga_core_function__ const_reference_type operator[](size_type i) const {
      return m_data[i];
    }

    __saga_core_function__ constexpr auto size() const { return m_size; }

  private:
    pointer_type m_data;
    size_type m_size;
  };

  /// Make an alterable view of a vector
  template <class Vector> auto make_vector_view(Vector &v) {
    return vector_view<typename Vector::value_type, Vector::backend>(v.data(),
                                                                     v.size());
  }

  /// Make a constant view of a vector
  template <class Vector> auto make_vector_view(Vector const &v) {
    return const_vector_view<typename Vector::value_type, Vector::backend>(
        v.data(), v.size());
  }

  namespace detail {

    template <class TypeDescriptor, class Fields> struct container_view_base;

    template <class TypeDescriptor, template <class> class... Field>
    struct container_view_base<TypeDescriptor, saga::properties<Field...>> {
      using type = saga::core::tuple<vector_view<
          saga::core::underlying_value_type_t<Field<TypeDescriptor>>,
          TypeDescriptor::backend>...>;
    };

    template <class TypeDescriptor, class Fields>
    using container_view_base_t =
        typename container_view_base<TypeDescriptor, Fields>::type;

    template <class TypeDescriptor, class Fields>
    struct const_container_view_base;

    template <class TypeDescriptor, template <class> class... Field>
    struct const_container_view_base<TypeDescriptor,
                                     saga::properties<Field...>> {
      using type = saga::core::tuple<const_vector_view<
          saga::core::underlying_value_type_t<Field<TypeDescriptor>>,
          TypeDescriptor::backend>...>;
    };

    template <class TypeDescriptor, class Fields>
    using const_container_view_base_t =
        typename const_container_view_base<TypeDescriptor, Fields>::type;

    template <class Container, template <class> class... Field>
    auto make_vector_views_impl(Container &other, saga::properties<Field...>) {

      using return_type =
          container_view_base_t<typename Container::type_descriptor,
                                saga::properties<Field...>>;

      return return_type{make_vector_view(other.template get_non_const<Field>())...};
    }

    template <class Container, template <class> class... Field>
    auto make_vector_views_impl(Container const &other,
                                saga::properties<Field...>) {

      using return_type =
          const_container_view_base_t<typename Container::type_descriptor,
                                      saga::properties<Field...>>;

      return return_type{make_vector_view(other.template get<Field>())...};
    }

    template <class Container> auto make_vector_views(Container &other) {
      return make_vector_views_impl(other, typename Container::fields_type{});
    }

    template <class Container> auto make_vector_views(Container const &other) {
      return make_vector_views_impl(other, typename Container::fields_type{});
    }
  } // namespace detail

  /// View of a container that can or can not be modified
  template <class Container>
  class container_view
      : detail::container_view_base_t<typename Container::type_descriptor,
                                      typename Container::fields_type> {

  public:
    using base_type =
        detail::container_view_base_t<typename Container::type_descriptor,
                                      typename Container::fields_type>;
    using type_descriptor = typename Container::type_descriptor;
    using fields_type = typename Container::fields_type;
    using value_type = typename Container::value<container_view>;
    using proxy_type = typename Container::proxy<container_view>;
    using const_proxy_type = typename Container::const_proxy<container_view>;
    using size_type = std::size_t;

    container_view(Container &container)
        : base_type{detail::make_vector_views(container)} {}

    container_view() = delete;
    container_view(container_view const &) = default;
    container_view(container_view &&) = default;
    container_view &operator=(container_view const &) = default;
    container_view &operator=(container_view &&) = default;

    __saga_core_function__ auto operator[](size_type i) {
      return proxy_type(this, i);
    }

    __saga_core_function__ auto operator[](size_type i) const {
      return const_proxy_type(this, i);
    }

    __saga_core_function__ constexpr auto size() const {
      return saga::core::get<0>(*this).size();
    }

    template <template <class> class F>
    __saga_core_function__ auto const &get() const {
      return saga::core::get<saga::core::property_index_v<F, fields_type>>(*this);
    }

    /// Get the value associated to the given field and index in the container
    template <template <class> class F>
    __saga_core_function__ auto &get(size_type i) {
      return this->template get<F>()[i];
    }
    /// Get the value associated to the given field and index in the container
    template <template <class> class F>
    __saga_core_function__ auto const &get(size_type i) const {
      return this->template get<F>()[i];
    }
    /// Set the value associated to the given field and index in the container
    template <template <class> class F>
    __saga_core_function__ void
    set(size_type i,
        saga::core::underlying_value_type_t<F<type_descriptor>> v) {
      saga::core::get<saga::core::property_index_v<F, fields_type>>(*this)[i] = v;
    }
  };

  /// View of a container that can not be modified
  template <class Container>
  class const_container_view
      : detail::const_container_view_base_t<typename Container::type_descriptor,
                                            typename Container::fields_type> {

  public:
    using base_type =
        detail::const_container_view_base_t<typename Container::type_descriptor,
                                      typename Container::fields_type>;
    using type_descriptor = typename Container::type_descriptor;
    using fields_type = typename Container::fields_type;
    using value_type = typename Container::value<const_container_view>;
    using proxy_type = typename Container::proxy<const_container_view>;
    using const_proxy_type =
        typename Container::const_proxy<const_container_view>;
    using size_type = std::size_t;

    const_container_view(Container &container)
        : base_type{detail::make_vector_views(container)} {}

    const_container_view() = delete;
    const_container_view(const_container_view const &) = default;
    const_container_view(const_container_view &&) = default;
    const_container_view &operator=(const_container_view const &) = default;
    const_container_view &operator=(const_container_view &&) = default;

    __saga_core_function__ auto operator[](size_type i) {
      return proxy_type(this, i);
    }

    __saga_core_function__ auto operator[](size_type i) const {
      return const_proxy_type(this, i);
    }

    __saga_core_function__ constexpr auto size() const {
      return saga::core::get<0>(*this).size();
    }

    template <template <class> class F>
    __saga_core_function__ auto const &get() const {
      return saga::core::get<saga::core::property_index_v<F, fields_type>>(*this);
    }
    /// Get the value associated to the given field and index in the container
    template <template <class> class F>
    __saga_core_function__ auto const &get(size_type i) const {
      return this->template get<F>()[i];
    }
  };

  template <class Container> auto make_container_view(Container &container) {
    return container_view(container);
  }

  template <class Container>
  auto make_container_view(Container const &container) {
    return const_container_view(container);
  }
} // namespace saga::core
