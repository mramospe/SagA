#pragma once
#include "saga/core/container.hpp"
#include "saga/core/fields.hpp"
#include "saga/core/types.hpp"
#include "saga/physics/quantities.hpp"

namespace saga::physics {

  template <class TypeDescriptor>
  class forces : public saga::core::container_with_fields<
                     TypeDescriptor, saga::property::x, saga::property::y,
                     saga::property::z, saga::property::t> {

  public:
    /// Base type
    using base_type =
        saga::core::container_with_fields<TypeDescriptor, saga::property::x,
                                          saga::property::y, saga::property::z,
                                          saga::property::t>;
    /// Constructors inherited from the base class
    using base_type::base_type;

    using float_type = typename TypeDescriptor::float_type;

    auto const &get_x() const {
      return this->template get<saga::property::x>();
    }
    auto &get_x() { return this->template get<saga::property::x>(); }
    void set_x(std::size_t i, float_type v) {
      this->template set<saga::property::x>(i, v);
    }
    auto const &get_y() const {
      return this->template get<saga::property::y>();
    }
    auto &get_y() { return this->template get<saga::property::y>(); }
    void set_y(std::size_t i, float_type v) {
      this->template set<saga::property::y>(i, v);
    }
    auto const &get_z() const {
      return this->template get<saga::property::z>();
    }
    auto &get_z() { return this->template get<saga::property::z>(); }
    void set_z(std::size_t i, float_type v) {
      this->template set<saga::property::z>(i, v);
    }
    auto const &get_t() const {
      return this->template get<saga::property::t>();
    }
    auto &get_t() { return this->template get<saga::property::t>(); }
    void set_t(std::size_t i, float_type v) {
      this->template set<saga::property::t>(i, v);
    }

    struct value_type : base_type::value_type {

      using base_type::value_type::value_type;

      auto const &get_x() const {
        return this->template get<saga::property::x>();
      }
      auto &get_x() { return this->template get<saga::property::x>(); }
      void set_x(float_type v) { this->template set<saga::property::x>(v); }
      auto const &get_y() const {
        return this->template get<saga::property::y>();
      }
      auto &get_y() { return this->template get<saga::property::y>(); }
      void set_y(float_type v) { this->template set<saga::property::y>(v); }
      auto const &get_z() const {
        return this->template get<saga::property::z>();
      }
      auto &get_z() { return this->template get<saga::property::z>(); }
      void set_z(float_type v) { this->template set<saga::property::z>(v); }
      auto const &get_t() const {
        return this->template get<saga::property::t>();
      }
      auto &get_t() { return this->template get<saga::property::t>(); }
      void set_t(float_type v) { this->template set<saga::property::t>(v); }
    };

    struct proxy_type : base_type::proxy_type {

      using base_type::proxy_type::proxy_type;

      proxy_type &operator*() { return *this; }

      proxy_type const &operator*() const { return *this; }

      proxy_type &operator++() {
        base_type::proxy_type::operator++();
        ;
        return *this;
      }

      proxy_type operator++(int) {

        auto copy = *this;
        base_type::proxy_type::operator++();
        return copy;
      }

      proxy_type &operator--() {
        base_type::proxy_type::operator--();
        return *this;
      }

      proxy_type operator--(int) {

        auto copy = *this;
        base_type::proxy_type::operator--();
        return copy;
      }

      auto const &get_x() const {
        return this->template get<saga::property::x>();
      }
      auto &get_x() { return this->template get<saga::property::x>(); }
      void set_x(float_type v) {
        this->container().template set<saga::property::x>(this->index(), v);
      }
      auto const &get_y() const {
        return this->template get<saga::property::y>();
      }
      auto &get_y() { return this->template get<saga::property::y>(); }
      void set_y(float_type v) {
        this->container().template set<saga::property::y>(this->index(), v);
      }
      auto const &get_z() const {
        return this->template get<saga::property::z>();
      }
      auto &get_z() { return this->template get<saga::property::z>(); }
      void set_z(float_type v) {
        this->container().template set<saga::property::z>(this->index(), v);
      }
      auto const &get_t() const {
        return this->template get<saga::property::t>();
      }
      auto &get_t() { return this->template get<saga::property::t>(); }
      void set_t(float_type v) {
        this->container().template set<saga::property::t>(this->index(), v);
      }
    };

    struct const_proxy_type : base_type::const_proxy_type {

      using base_type::const_proxy_type::const_proxy_type;

      const_proxy_type &operator*() { return *this; }

      const_proxy_type const &operator*() const { return *this; }

      const_proxy_type &operator++() {
        base_type::const_proxy_type::operator++();
        ;
        return *this;
      }

      const_proxy_type operator++(int) {

        auto copy = *this;
        base_type::const_proxy_type::operator++();
        return copy;
      }

      const_proxy_type &operator--() {
        base_type::proxy_type::operator--();
        return *this;
      }

      const_proxy_type operator--(int) {

        auto copy = *this;
        base_type::const_proxy_type::operator--();
        return copy;
      }

      auto const &get_x() const {
        return this->template get<saga::property::x>();
      }
      auto &get_x() { return this->template get<saga::property::x>(); }
      auto const &get_y() const {
        return this->template get<saga::property::y>();
      }
      auto &get_y() { return this->template get<saga::property::y>(); }
      auto const &get_z() const {
        return this->template get<saga::property::z>();
      }
      auto &get_z() { return this->template get<saga::property::z>(); }
      auto const &get_t() const {
        return this->template get<saga::property::t>();
      }
      auto &get_t() { return this->template get<saga::property::t>(); }
    };

    friend bool operator==(proxy_type const &f, proxy_type const &s) {
      return (f.container_ptr() == s.container_ptr()) &&
             (f.index() == s.index());
    }

    friend bool operator==(const_proxy_type const &f,
                           const_proxy_type const &s) {
      return (f.container_ptr() == s.container_ptr()) &&
             (f.index() == s.index());
    }

    friend bool operator==(const_proxy_type const &f, proxy_type const &s) {
      return (f.container_ptr() == s.container_ptr()) &&
             (f.index() == s.index());
    }

    friend bool operator==(proxy_type const &f, const_proxy_type const &s) {
      return (f.container_ptr() == s.container_ptr()) &&
             (f.index() == s.index());
    }

    friend bool operator!=(proxy_type const &f, proxy_type const &s) {
      return !(f == s);
    }

    friend bool operator!=(const_proxy_type const &f,
                           const_proxy_type const &s) {
      return !(f == s);
    }

    friend bool operator!=(const_proxy_type const &f, proxy_type const &s) {
      return !(f == s);
    }

    friend bool operator!=(proxy_type const &f, const_proxy_type const &s) {
      return !(f == s);
    }

    /// Access an element of the container
    auto operator[](std::size_t idx) { return proxy_type(*this, idx); }

    /// Access an element of the container
    auto operator[](std::size_t idx) const {

      return const_proxy_type(*this, idx);
    }

    /// Begining of the container
    auto begin() { return proxy_type(*this, 0); }

    /// Begining of the container
    auto begin() const { return const_proxy_type(*this, 0); }

    /// Begining of the container
    auto cbegin() const { return const_proxy_type(*this, 0); }

    /// End of the container
    auto end() { return proxy_type(*this, this->size()); }

    /// End of the container
    auto end() const { return const_proxy_type(*this, this->size()); }

    /// End of the container
    auto cend() const { return const_proxy_type(*this, this->size()); }
  };
} // namespace saga::physics
