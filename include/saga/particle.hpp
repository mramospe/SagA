#pragma once
#include "saga/core/container.hpp"
#include "saga/core/fields.hpp"
#include "saga/core/types.hpp"
#include "saga/physics/quantities.hpp"
#include "saga/physics/shape.hpp"
#include <cmath>

namespace saga {

  namespace detail {

    template <class TypeDescriptor, class Properties>
    struct container_with_fields_from_properties;

    template <class TypeDescriptor, template <class> class... Property>
    struct container_with_fields_from_properties<
        TypeDescriptor, saga::properties<Property...>> {
      using type =
          saga::core::container_with_fields<TypeDescriptor, Property...>;
    };

    template <class TypeDescriptor, class Properties>
    using container_with_fields_from_properties_t =
        typename container_with_fields_from_properties<TypeDescriptor,
                                                       Properties>::type;

    template <class TypeDescriptor, template <class> class Shape,
              template <class> class... Property>
    using container_with_fields_from_properties_and_shape_t =
        container_with_fields_from_properties_t<
            TypeDescriptor,
            saga::append_properties_t<
                typename Shape<TypeDescriptor>::properties, Property...>>;

    /// Base type for a container of particles
    template <class TypeDescriptor, template <class> class Shape,
              template <class> class... Property>
    using particle_container_base_type =
        detail::container_with_fields_from_properties_and_shape_t<
            TypeDescriptor, Shape, saga::property::x, saga::property::y,
            saga::property::z, saga::property::t, saga::property::px,
            saga::property::py, saga::property::pz, saga::property::e,
            Property...>;

    /// Container of particles
    template <class TypeDescriptor,
              template <class> class Shape = saga::physics::point,
              class Properties = saga::properties<>>
    class particle_container;

    /// Container of particles
    template <class TypeDescriptor, template <class T> class Shape,
              template <class T> class... Property>
    class particle_container<TypeDescriptor, Shape,
                             saga::properties<Property...>>
        : public particle_container_base_type<TypeDescriptor, Shape,
                                              Property...> {
    public:
      /// Base type
      using base_type =
          particle_container_base_type<TypeDescriptor, Shape, Property...>;
      /// Constructors inherited from the base class
      using base_type::base_type;
      /// Shape type
      using shape_type = Shape<TypeDescriptor>;

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

      class value_type;
      class proxy_type;
      class const_proxy_type;

      class value_type : public base_type::value_type {

      public:
        using base_type::value_type::value_type;
        using shape_type = particle_container::shape_type;

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
          set_px(std::abs(get_px()) >
                         saga::types::numeric_info<TypeDescriptor>::min
                     ? v * get_px() / mass
                     : 0.f);
          set_py(std::abs(get_py()) >
                         saga::types::numeric_info<TypeDescriptor>::min
                     ? v * get_py() / mass
                     : 0.f);
          set_pz(std::abs(get_pz()) >
                         saga::types::numeric_info<TypeDescriptor>::min
                     ? v * get_pz() / mass
                     : 0.f);
          set_e(std::sqrt(v * v + get_px() * get_px() + get_py() * get_py() +
                          get_pz() * get_pz()));
        }
      };

      class proxy_type : public base_type::proxy_type {

      public:
        using container_type = particle_container;
        using shape_type = container_type::shape_type;

        proxy_type(container_type &cont, std::size_t idx)
            : base_type::proxy_type(cont, idx) {}
        proxy_type(proxy_type &&other)
            : base_type::proxy_type(other.container(), other.index()) {}
        proxy_type(proxy_type const &other)
            : base_type::proxy_type(other.container(), other.index()) {}

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
          set_px(std::abs(get_px()) >
                         saga::types::numeric_info<TypeDescriptor>::min
                     ? v * get_px() / mass
                     : 0.f);
          set_py(std::abs(get_py()) >
                         saga::types::numeric_info<TypeDescriptor>::min
                     ? v * get_py() / mass
                     : 0.f);
          set_pz(std::abs(get_pz()) >
                         saga::types::numeric_info<TypeDescriptor>::min
                     ? v * get_pz() / mass
                     : 0.f);
          set_e(std::sqrt(v * v + get_px() * get_px() + get_py() * get_py() +
                          get_pz() * get_pz()));
        }
      };

      class const_proxy_type : public base_type::const_proxy_type {
      public:
        using container_type = particle_container;
        using shape_type = container_type::shape_type;

        const_proxy_type(container_type const &cont, std::size_t idx)
            : base_type::const_proxy_type(cont, idx) {}
        const_proxy_type(const_proxy_type &&other)
            : base_type::const_proxy_type(other.container(), other.index()) {}
        const_proxy_type(const_proxy_type const &other)
            : base_type::const_proxy_type(other.container(), other.index()) {}

        const_proxy_type &operator*() { return *this; }

        const_proxy_type const &operator*() const { return *this; }

        const_proxy_type &operator++() {
          base_type::const_proxy_type::operator++();
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
  } // namespace detail

  /// Template for particles
  template <class TypeDescriptor,
            template <class> class Shape = saga::physics::point,
            class Properties = saga::properties<>>
  using particles =
      saga::detail::particle_container<TypeDescriptor, Shape, Properties>;

  /// Standard template for individual particles
  template <class TypeDescriptor,
            template <class> class Shape = saga::physics::point,
            class Properties = saga::properties<>>
  using particle =
      typename particles<TypeDescriptor, Shape, Properties>::value_type;

} // namespace saga
