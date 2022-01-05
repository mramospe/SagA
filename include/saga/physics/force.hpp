#pragma once
#include "saga/core/container.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/types.hpp"

namespace saga::physics {

  /*!\brief Container of forces
   */
  template <class TypeDescriptor>
  class forces : public saga::core::container_with_fields<
                     TypeDescriptor, saga::property::x, saga::property::y,
                     saga::property::z> {

  public:
    /// Base type
    using base_type =
        saga::core::container_with_fields<TypeDescriptor, saga::property::x,
                                          saga::property::y, saga::property::z>;
    /// Constructors inherited from the base class
    using base_type::base_type;

    using float_type = typename TypeDescriptor::float_type;
    using type_descriptor = TypeDescriptor;

    // Forward declarations
    template <class ContainerOrView> class value;

    template <class ContainerOrView> class proxy;

    template <class ContainerOrView> class const_proxy;

    template <class ContainerOrView>
    class value : public saga::core::value<ContainerOrView> {
    public:
      using base_type = saga::core::value<ContainerOrView>;

      __saga_core_function__ value(saga::core::underlying_value_type_t<
                                       saga::property::x<type_descriptor>> &&vx,
                                   saga::core::underlying_value_type_t<
                                       saga::property::y<type_descriptor>> &&vy,
                                   saga::core::underlying_value_type_t<
                                       saga::property::z<type_descriptor>> &&vz)
          : base_type(std::forward<saga::core::underlying_value_type_t<
                          saga::property::x<type_descriptor>>>(vx),
                      std::forward<saga::core::underlying_value_type_t<
                          saga::property::y<type_descriptor>>>(vy),
                      std::forward<saga::core::underlying_value_type_t<
                          saga::property::z<type_descriptor>>>(vz)) {}
      __saga_core_function__
      value(saga::core::underlying_value_type_t<
                saga::property::x<type_descriptor>> const &vx,
            saga::core::underlying_value_type_t<
                saga::property::y<type_descriptor>> const &vy,
            saga::core::underlying_value_type_t<
                saga::property::z<type_descriptor>> const &vz)
          : base_type(vx, vy, vz) {}

      value() = default;
      value(value const &) = default;
      value(value &&) = default;
      value &operator=(value &&) = default;
      value &operator=(value const &) = default;

      __saga_core_function__ value &operator=(proxy<ContainerOrView> const &p) {
        base_type::operator=(p);
        return *this;
      }
      __saga_core_function__ value &
      operator=(const_proxy<ContainerOrView> const &p) {
        base_type::operator=(p);
        return *this;
      }

      __saga_core_function__ auto const &get_x() const {
        return this->template get<saga::property::x>();
      }
      __saga_core_function__ auto &get_x() {
        return this->template get<saga::property::x>();
      }
      __saga_core_function__ void set_x(float_type v) {
        this->template set<saga::property::x>(v);
      }
      __saga_core_function__ auto const &get_y() const {
        return this->template get<saga::property::y>();
      }
      __saga_core_function__ auto &get_y() {
        return this->template get<saga::property::y>();
      }
      __saga_core_function__ void set_y(float_type v) {
        this->template set<saga::property::y>(v);
      }
      __saga_core_function__ auto const &get_z() const {
        return this->template get<saga::property::z>();
      }
      __saga_core_function__ auto &get_z() {
        return this->template get<saga::property::z>();
      }
      __saga_core_function__ void set_z(float_type v) {
        this->template set<saga::property::z>(v);
      }
    };

    template <class ContainerOrView>
    class proxy : public saga::core::proxy<ContainerOrView> {
    public:
      using base_type = saga::core::proxy<ContainerOrView>;
      using container_type = ContainerOrView;
      using container_pointer_type = container_type *;
      using size_type = typename base_type::size_type;

      proxy() = delete;
      proxy(proxy const &) = default;
      proxy(proxy &) = default;
      __saga_core_function__ proxy(container_pointer_type cont, size_type idx)
          : base_type{cont, idx} {}

      proxy &operator=(proxy const &) = default;
      proxy &operator=(proxy &&) = default;

      __saga_core_function__ proxy &operator=(value<ContainerOrView> const &p) {
        base_type::operator=(p);
        return *this;
      }

      __saga_core_function__ proxy &operator=(value<ContainerOrView> &&p) {
        base_type::operator=(p);
        return *this;
      }

      __saga_core_function__ auto const &get_x() const {
        return this->template get<saga::property::x>();
      }
      __saga_core_function__ auto &get_x() {
        return this->template get<saga::property::x>();
      }
      __saga_core_function__ void set_x(float_type v) {
        this->container().template set<saga::property::x>(this->index(), v);
      }
      __saga_core_function__ auto const &get_y() const {
        return this->template get<saga::property::y>();
      }
      __saga_core_function__ auto &get_y() {
        return this->template get<saga::property::y>();
      }
      __saga_core_function__ void set_y(float_type v) {
        this->container().template set<saga::property::y>(this->index(), v);
      }
      __saga_core_function__ auto const &get_z() const {
        return this->template get<saga::property::z>();
      }
      __saga_core_function__ auto &get_z() {
        return this->template get<saga::property::z>();
      }
      __saga_core_function__ void set_z(float_type v) {
        this->container().template set<saga::property::z>(this->index(), v);
      }
    };

    template <class ContainerOrView>
    class const_proxy : public saga::core::const_proxy<ContainerOrView> {
    public:
      using base_type = saga::core::const_proxy<ContainerOrView>;
      using container_type = ContainerOrView;
      using container_pointer_type = container_type const *;
      using size_type = typename base_type::size_type;

      const_proxy() = delete;
      const_proxy(const_proxy const &) = default;
      const_proxy(const_proxy &) = default;
      __saga_core_function__ const_proxy(container_pointer_type cont,
                                         size_type idx)
          : base_type{cont, idx} {}

      const_proxy &operator=(const_proxy const &p) = default;
      const_proxy &operator=(const_proxy &&p) = default;

      __saga_core_function__ auto const &get_x() const {
        return this->template get<saga::property::x>();
      }
      __saga_core_function__ auto &get_x() {
        return this->template get<saga::property::x>();
      }
      __saga_core_function__ auto const &get_y() const {
        return this->template get<saga::property::y>();
      }
      __saga_core_function__ auto &get_y() {
        return this->template get<saga::property::y>();
      }
      __saga_core_function__ auto const &get_z() const {
        return this->template get<saga::property::z>();
      }
      __saga_core_function__ auto &get_z() {
        return this->template get<saga::property::z>();
      }
    };

    using value_type = value<forces>;
    using proxy_type = proxy<forces>;
    using const_proxy_type = const_proxy<forces>;
    using iterator_type = saga::core::proxy_iterator<forces>;
    using const_iterator_type = saga::core::const_proxy_iterator<forces>;

    /// Access an element of the container
    __saga_core_function__ auto operator[](std::size_t idx) {
      return proxy_type(this, idx);
    }

    /// Access an element of the container
    __saga_core_function__ auto operator[](std::size_t idx) const {

      return const_proxy_type(this, idx);
    }

    /// Begining of the container
    auto begin() { return iterator_type(this, 0); }

    /// Begining of the container
    auto begin() const { return const_iterator_type(this, 0); }

    /// Begining of the container
    auto cbegin() const { return const_iterator_type(this, 0); }

    /// End of the container
    auto end() { return iterator_type(this, this->size()); }

    /// End of the container
    auto end() const { return const_iterator_type(this, this->size()); }

    /// End of the container
    auto cend() const { return const_iterator_type(this, this->size()); }
  };
} // namespace saga::physics
