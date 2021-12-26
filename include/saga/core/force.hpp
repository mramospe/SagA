#pragma once
#include "saga/core/container.hpp"
#include "saga/core/fields.hpp"
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

    // Forward declarations
    class value_type;
    class iterator_type;
    class proxy_type;
    class const_iterator_type;
    class const_proxy_type;

    class value_type : public base_type::value_type {
    public:
      using base_type::value_type::value_type;

      value_type &operator=(proxy_type const &p) {
        base_type::value_type::operator=(p);
        return *this;
      }
      value_type &operator=(const_proxy_type const &p) {
        base_type::value_type::operator=(p);
        return *this;
      }

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
    };

    class proxy_type : public base_type::proxy_type {
    public:
      using container_type = forces;
      using container_pointer_type = container_type *;

      /// Build the proxy from the container and the index
      proxy_type(container_pointer_type cont, std::size_t idx)
          : base_type::proxy_type(cont, idx) {}
      proxy_type(proxy_type &&other)
          : base_type::proxy_type(
                static_cast<container_pointer_type>(other.container_ptr()),
                other.index()) {}
      proxy_type(proxy_type const &other)
          : base_type::proxy_type(
                static_cast<container_pointer_type>(other.container_ptr()),
                other.index()) {}

      proxy_type &operator=(proxy_type const &p) {
        base_type::proxy_type::operator=(p);
        return *this;
      }

      proxy_type &operator=(proxy_type &&p) {
        base_type::proxy_type::operator=(p);
        return *this;
      }

      proxy_type &operator=(value_type const &p) {
        base_type::proxy_type::operator=(p);
        return *this;
      }

      proxy_type &operator=(value_type &&p) {
        base_type::proxy_type::operator=(p);
        return *this;
      }

      proxy_type &operator++() {
        base_type::proxy_type::operator++();
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
    };

    class iterator_type : public base_type::iterator_type {
    public:
      using container_type = forces;
      using container_pointer_type = container_type *;

      /// Build the proxy from the container and the index
      iterator_type(container_pointer_type cont, std::size_t idx)
          : base_type::iterator_type(cont, idx) {}
      iterator_type(iterator_type &&other) = default;
      iterator_type(iterator_type const &) = default;
      iterator_type &operator=(iterator_type const &) = default;
      iterator_type &operator=(iterator_type &&) = default;

      proxy_type operator*() {
        return proxy_type{
            static_cast<container_pointer_type>(this->container_ptr()),
            this->index()};
      }
      const_proxy_type operator*() const {
        return const_proxy_type{
            static_cast<container_pointer_type>(this->container_ptr()),
            this->index()};
      }

      iterator_type &operator++() {
        base_type::iterator_type::operator++();
        return *this;
      }

      iterator_type operator++(int) {

        auto copy = *this;
        base_type::iterator_type::operator++();
        return copy;
      }

      iterator_type &operator--() {
        base_type::iterator_type::operator--();
        return *this;
      }

      iterator_type operator--(int) {

        auto copy = *this;
        base_type::iterator_type::operator--();
        return copy;
      }

      iterator_type &operator+=(int i) {
        base_type::iterator_type::operator+=(i);
        return *this;
      }

      iterator_type operator-=(int i) {
        base_type::iterator_type::operator-=(i);
        return *this;
      }

      iterator_type operator+(int i) {
        auto copy = *this;
        copy += i;
        return copy;
      }

      iterator_type operator-(int i) {
        auto copy = *this;
        copy -= i;
        return copy;
      }
    };

    class const_proxy_type : public base_type::const_proxy_type {
    public:
      using container_type = forces;
      using container_pointer_type = container_type const *;

      const_proxy_type(container_pointer_type cont, std::size_t idx)
          : base_type::const_proxy_type(cont, idx) {}
      const_proxy_type(const_proxy_type &&other)
          : base_type::const_proxy_type(other.container_ptr(), other.index()) {}
      const_proxy_type(const_proxy_type const &other)
          : base_type::const_proxy_type(other.container_ptr(), other.index()) {}

      const_proxy_type &operator=(const_proxy_type const &p) {
        base_type::const_proxy_type::operator=(p);
        return *this;
      }

      const_proxy_type &operator=(const_proxy_type &&p) {
        base_type::const_proxy_type::operator=(p);
        return *this;
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
    };

    class const_iterator_type : public base_type::const_iterator_type {
    public:
      using container_type = forces;
      using container_pointer_type = container_type const *;

      const_iterator_type(container_pointer_type cont, std::size_t idx)
          : base_type::const_iterator_type(cont, idx) {}
      const_iterator_type(const_iterator_type &&other) = default;
      const_iterator_type(const_iterator_type const &other) = default;

      const_iterator_type &operator=(const_iterator_type const &) = default;
      const_iterator_type &operator=(const_iterator_type &&) = default;

      const_proxy_type operator*() {
        return const_proxy_type{
            static_cast<container_pointer_type>(this->container_ptr()),
            this->index()};
      }

      const_proxy_type operator*() const {
        return const_proxy_type{
            static_cast<container_pointer_type>(this->container_ptr()),
            this->index()};
      }

      const_iterator_type &operator++() {
        base_type::const_iterator_type::operator++();
        return *this;
      }

      const_iterator_type operator++(int) {

        auto copy = *this;
        base_type::const_iterator_type::operator++();
        return copy;
      }

      const_iterator_type &operator--() {
        base_type::iterator_type::operator--();
        return *this;
      }

      const_iterator_type operator--(int) {

        auto copy = *this;
        base_type::const_iterator_type::operator--();
        return copy;
      }

      const_iterator_type &operator+=(int i) {
        base_type::const_iterator_type::operator+=(i);
        return *this;
      }

      const_iterator_type operator-=(int i) {
        base_type::const_iterator_type::operator-=(i);
        return *this;
      }

      const_iterator_type operator+(int i) {
        auto copy = *this;
        copy += i;
        return copy;
      }

      const_iterator_type operator-(int i) {
        auto copy = *this;
        copy -= i;
        return copy;
      }
    };

    /// Access an element of the container
    auto operator[](std::size_t idx) { return proxy_type(this, idx); }

    /// Access an element of the container
    auto operator[](std::size_t idx) const {

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
