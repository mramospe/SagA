#pragma once
#include "saga/core/container.hpp"
#include "saga/core/fields.hpp"
#include "saga/core/types.hpp"
#include "saga/physics/quantities.hpp"
#include <cmath>

namespace saga {

  namespace core {

    /// Base type for a container of particles
    template <class TypeDescriptor, template <class> class... Property>
    using particle_container_base_type = container_with_fields<
        TypeDescriptor, saga::property::x, saga::property::y, saga::property::z,
        saga::property::t, saga::property::px, saga::property::py,
        saga::property::pz, saga::property::e, Property...>;

    /// Container of particles
    template <class TypeDescriptor, class Properties> class particle_container;

    /// Container of particles
    template <class TypeDescriptor, template <class T> class... Property>
    class particle_container<TypeDescriptor, saga::properties<Property...>>
        : public particle_container_base_type<TypeDescriptor, Property...> {
    public:
      /// Base type
      using base_type =
          particle_container_base_type<TypeDescriptor, Property...>;
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
      auto const &get_px() const {
        return this->template get<saga::property::px>();
      }
      auto &get_px() { return this->template get<saga::property::px>(); }
      void set_px(std::size_t i, float_type v) {
        this->template set<saga::property::px>(i, v);
      }
      auto const &get_py() const {
        return this->template get<saga::property::py>();
      }
      auto &get_py() { return this->template get<saga::property::py>(); }
      void set_py(std::size_t i, float_type v) {
        this->template set<saga::property::py>(i, v);
      }
      auto const &get_pz() const {
        return this->template get<saga::property::pz>();
      }
      auto &get_pz() { return this->template get<saga::property::pz>(); }
      void set_pz(std::size_t i, float_type v) {
        this->template set<saga::property::pz>(i, v);
      }
      auto const &get_e() const {
        return this->template get<saga::property::e>();
      }
      auto &get_e() { return this->template get<saga::property::e>(); }
      void set_e(std::size_t i, float_type v) {
        this->template set<saga::property::e>(i, v);
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
        auto const &get_px() const {
          return this->template get<saga::property::px>();
        }
        auto &get_px() { return this->template get<saga::property::px>(); }
        void set_px(float_type v) { this->template set<saga::property::px>(v); }
        auto const &get_py() const {
          return this->template get<saga::property::py>();
        }
        auto &get_py() { return this->template get<saga::property::py>(); }
        void set_py(float_type v) { this->template set<saga::property::py>(v); }
        auto const &get_pz() const {
          return this->template get<saga::property::pz>();
        }
        auto &get_pz() { return this->template get<saga::property::pz>(); }
        void set_pz(float_type v) { this->template set<saga::property::pz>(v); }
        auto const &get_e() const {
          return this->template get<saga::property::e>();
        }
        auto &get_e() { return this->template get<saga::property::e>(); }
        void set_e(float_type v) { this->template set<saga::property::e>(v); }
        auto get_mass() const {
          return std::sqrt(std::abs(get_e() * get_e() - get_px() * get_px() -
                                    get_py() * get_py() - get_pz() * get_pz()));
        }
        void set_mass(float_type v) {
          auto mass = get_mass();
          set_px(v * get_px() / mass);
          set_py(v * get_py() / mass);
          set_pz(v * get_pz() / mass);
          set_e(std::sqrt(v * v + get_px() * get_px() + get_py() * get_py() +
                          get_pz() * get_pz()));
        }
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
        auto const &get_px() const {
          return this->template get<saga::property::px>();
        }
        auto &get_px() { return this->template get<saga::property::px>(); }
        void set_px(float_type v) {
          this->container().template set<saga::property::px>(this->index(), v);
        }
        auto const &get_py() const {
          return this->template get<saga::property::py>();
        }
        auto &get_py() { return this->template get<saga::property::py>(); }
        void set_py(float_type v) {
          this->container().template set<saga::property::py>(this->index(), v);
        }
        auto const &get_pz() const {
          return this->template get<saga::property::pz>();
        }
        auto &get_pz() { return this->template get<saga::property::pz>(); }
        void set_pz(float_type v) {
          this->container().template set<saga::property::pz>(this->index(), v);
        }
        auto const &get_e() const {
          return this->template get<saga::property::e>();
        }
        auto &get_e() { return this->template get<saga::property::e>(); }
        void set_e(float_type v) {
          this->container().template set<saga::property::e>(this->index(), v);
        }
        auto get_mass() const {
          return std::sqrt(std::abs(get_e() * get_e() - get_px() * get_px() -
                                    get_py() * get_py() - get_pz() * get_pz()));
        }
        void set_mass(float_type v) {
          auto mass = get_mass();
          set_px(v * get_px() / mass);
          set_py(v * get_py() / mass);
          set_pz(v * get_pz() / mass);
          set_e(std::sqrt(v * v + get_px() * get_px() + get_py() * get_py() +
                          get_pz() * get_pz()));
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
        auto const &get_px() const {
          return this->template get<saga::property::px>();
        }
        auto &get_px() { return this->template get<saga::property::px>(); }
        auto const &get_py() const {
          return this->template get<saga::property::py>();
        }
        auto &get_py() { return this->template get<saga::property::py>(); }
        auto const &get_pz() const {
          return this->template get<saga::property::pz>();
        }
        auto &get_pz() { return this->template get<saga::property::pz>(); }
        auto const &get_e() const {
          return this->template get<saga::property::e>();
        }
        auto &get_e() { return this->template get<saga::property::e>(); }
        auto get_mass() const {
          return std::sqrt(std::abs(get_e() * get_e() - get_px() * get_px() -
                                    get_py() * get_py() - get_pz() * get_pz()));
        }
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
  } // namespace core

  /// Template for particles
  template <class TypeDescriptor, class Properties = saga::properties<>>
  using particles = saga::core::particle_container<TypeDescriptor, Properties>;

  /// Standard template for individual particles
  template <class TypeDescriptor, class Properties = saga::properties<>>
  using particle = typename particles<TypeDescriptor, Properties>::value_type;

} // namespace saga
