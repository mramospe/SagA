#pragma once
#include "saga/core/container.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/types.hpp"

namespace saga::core {

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
    class value_type;
    class proxy_type;
    class const_proxy_type;

    class value_type : public saga::core::value<forces> {
    public:
      using base_type = saga::core::value<forces>;

      __saga_core_function__
      value_type(saga::core::underlying_value_type_t<
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
      value_type(saga::core::underlying_value_type_t<
                     saga::property::x<type_descriptor>> const &vx,
                 saga::core::underlying_value_type_t<
                     saga::property::y<type_descriptor>> const &vy,
                 saga::core::underlying_value_type_t<
                     saga::property::z<type_descriptor>> const &vz)
          : base_type(vx, vy, vz) {}

      value_type() = default;
      value_type(value_type const &) = default;
      value_type(value_type &&) = default;
      value_type &operator=(value_type &&) = default;
      value_type &operator=(value_type const &) = default;

      __saga_core_function__ value_type &operator=(proxy_type const &p) {
        base_type::operator=(p);
        return *this;
      }
      __saga_core_function__ value_type &operator=(const_proxy_type const &p) {
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

    class proxy_type : public saga::core::proxy<forces> {
    public:
      using base_type = saga::core::proxy<forces>;
      using container_type = forces;
      using container_pointer_type = container_type *;
      using size_type = typename base_type::size_type;

      proxy_type() = delete;
      proxy_type(proxy_type const &) = default;
      proxy_type(proxy_type &) = default;
      __saga_core_function__ proxy_type(container_pointer_type cont,
                                        size_type idx)
          : base_type{cont, idx} {}

      proxy_type &operator=(proxy_type const &) = default;
      proxy_type &operator=(proxy_type &&) = default;

      __saga_core_function__ proxy_type &operator=(value_type const &p) {
        base_type::operator=(p);
        return *this;
      }

      __saga_core_function__ proxy_type &operator=(value_type &&p) {
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

    class const_proxy_type : public saga::core::const_proxy<forces> {
    public:
      using base_type = saga::core::const_proxy<forces>;
      using container_type = forces;
      using container_pointer_type = container_type const *;
      using size_type = typename base_type::size_type;

      const_proxy_type() = delete;
      const_proxy_type(const_proxy_type const &) = default;
      const_proxy_type(const_proxy_type &) = default;
      __saga_core_function__ const_proxy_type(container_pointer_type cont,
                                              size_type idx)
          : base_type{cont, idx} {}

      const_proxy_type &operator=(const_proxy_type const &p) = default;
      const_proxy_type &operator=(const_proxy_type &&p) = default;

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
} // namespace saga::core
