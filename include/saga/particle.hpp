#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/container.hpp"
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
            saga::core::append_properties_t<
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
      /// Type descriptor
      using type_descriptor = TypeDescriptor;
      /// Base type
      using base_type =
          particle_container_base_type<type_descriptor, Shape, Property...>;
      /// Constructors inherited from the base class
      using base_type::base_type;
      /// Shape type
      using shape_type = Shape<type_descriptor>;
      /// Floating-point type used in the calculations
      using float_type = typename type_descriptor::float_type;
      /// Alias to a type where the backend has been switched to a one different
      /// from the previous
      template <saga::backend NewBackend>
      using type_with_backend =
          particle_container<saga::core::change_type_descriptor_backend_t<
                                 NewBackend, type_descriptor>,
                             Shape, saga::properties<Property...>>;

      // forward declarations
      template <class ContainerOrView> class value;

      template <class ContainerOrView> class proxy;

      template <class ContainerOrView> class const_proxy;

      template <class ContainerOrView>
      class value : public saga::core::value<ContainerOrView> {

      public:
        using base_type = saga::core::value<ContainerOrView>;
        using shape_type = particle_container::shape_type;

        template <class... T>
        __saga_core_function__ value(T &&...v)
            : base_type{std::forward<T>(v)...} {}

        template <class... T>
        __saga_core_function__ value(T const &...v) : base_type{v...} {}

        value() = default;
        value(value const &) = default;
        value(value &&) = default;
        value &operator=(value &&) = default;
        value &operator=(value const &) = default;

        __saga_core_function__ value &
        operator=(proxy<ContainerOrView> const &p) {
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
        __saga_core_function__ auto const &get_t() const {
          return this->template get<saga::property::t>();
        }
        __saga_core_function__ auto &get_t() {
          return this->template get<saga::property::t>();
        }
        __saga_core_function__ void set_t(float_type v) {
          this->template set<saga::property::t>(v);
        }
        __saga_core_function__ auto const &get_px() const {
          return this->template get<saga::property::px>();
        }
        __saga_core_function__ auto &get_px() {
          return this->template get<saga::property::px>();
        }
        __saga_core_function__ void set_px(float_type v) {
          this->template set<saga::property::px>(v);
        }
        __saga_core_function__ auto const &get_py() const {
          return this->template get<saga::property::py>();
        }
        __saga_core_function__ auto &get_py() {
          return this->template get<saga::property::py>();
        }
        __saga_core_function__ void set_py(float_type v) {
          this->template set<saga::property::py>(v);
        }
        __saga_core_function__ auto const &get_pz() const {
          return this->template get<saga::property::pz>();
        }
        __saga_core_function__ auto &get_pz() {
          return this->template get<saga::property::pz>();
        }
        __saga_core_function__ void set_pz(float_type v) {
          this->template set<saga::property::pz>(v);
        }
        __saga_core_function__ auto const &get_e() const {
          return this->template get<saga::property::e>();
        }
        __saga_core_function__ auto &get_e() {
          return this->template get<saga::property::e>();
        }
        __saga_core_function__ void set_e(float_type v) {
          this->template set<saga::property::e>(v);
        }
        __saga_core_function__ auto get_mass() const {
          return std::sqrt(std::abs(get_e() * get_e() - get_px() * get_px() -
                                    get_py() * get_py() - get_pz() * get_pz()));
        }
        __saga_core_function__ void set_momenta_and_mass(float_type px,
                                                         float_type py,
                                                         float_type pz,
                                                         float_type mass) {
          set_px(px);
          set_py(py);
          set_pz(pz);
          set_e(std::sqrt(px * px + py * py + pz * pz + mass * mass));
        }
      };

      template <class ContainerOrView>
      class proxy : public saga::core::proxy<ContainerOrView> {

      public:
        using base_type = saga::core::proxy<ContainerOrView>;
        using shape_type = particle_container::shape_type;
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

        __saga_core_function__ proxy &
        operator=(value<ContainerOrView> const &p) {
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
        __saga_core_function__ auto const &get_t() const {
          return this->template get<saga::property::t>();
        }
        __saga_core_function__ auto &get_t() {
          return this->template get<saga::property::t>();
        }
        __saga_core_function__ void set_t(float_type v) {
          this->container().template set<saga::property::t>(this->index(), v);
        }
        __saga_core_function__ auto const &get_px() const {
          return this->template get<saga::property::px>();
        }
        __saga_core_function__ auto &get_px() {
          return this->template get<saga::property::px>();
        }
        __saga_core_function__ void set_px(float_type v) {
          this->container().template set<saga::property::px>(this->index(), v);
        }
        __saga_core_function__ auto const &get_py() const {
          return this->template get<saga::property::py>();
        }
        __saga_core_function__ auto &get_py() {
          return this->template get<saga::property::py>();
        }
        __saga_core_function__ void set_py(float_type v) {
          this->container().template set<saga::property::py>(this->index(), v);
        }
        __saga_core_function__ auto const &get_pz() const {
          return this->template get<saga::property::pz>();
        }
        __saga_core_function__ auto &get_pz() {
          return this->template get<saga::property::pz>();
        }
        __saga_core_function__ void set_pz(float_type v) {
          this->container().template set<saga::property::pz>(this->index(), v);
        }
        __saga_core_function__ auto const &get_e() const {
          return this->template get<saga::property::e>();
        }
        __saga_core_function__ auto &get_e() {
          return this->template get<saga::property::e>();
        }
        __saga_core_function__ void set_e(float_type v) {
          this->container().template set<saga::property::e>(this->index(), v);
        }
        __saga_core_function__ auto get_mass() const {
          return std::sqrt(std::abs(get_e() * get_e() - get_px() * get_px() -
                                    get_py() * get_py() - get_pz() * get_pz()));
        }
        __saga_core_function__ void set_momenta_and_mass(float_type px,
                                                         float_type py,
                                                         float_type pz,
                                                         float_type mass) {
          set_px(px);
          set_py(py);
          set_pz(pz);
          set_e(std::sqrt(px * px + py * py + pz * pz + mass * mass));
        }
      };

      template <class ContainerOrView>
      class const_proxy : public saga::core::const_proxy<ContainerOrView> {
      public:
        using base_type = saga::core::const_proxy<ContainerOrView>;
        using shape_type = particle_container::shape_type;
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
        __saga_core_function__ auto const &get_t() const {
          return this->template get<saga::property::t>();
        }
        __saga_core_function__ auto &get_t() {
          return this->template get<saga::property::t>();
        }
        __saga_core_function__ auto const &get_px() const {
          return this->template get<saga::property::px>();
        }
        __saga_core_function__ auto &get_px() {
          return this->template get<saga::property::px>();
        }
        __saga_core_function__ auto const &get_py() const {
          return this->template get<saga::property::py>();
        }
        __saga_core_function__ auto &get_py() {
          return this->template get<saga::property::py>();
        }
        __saga_core_function__ auto const &get_pz() const {
          return this->template get<saga::property::pz>();
        }
        __saga_core_function__ auto &get_pz() {
          return this->template get<saga::property::pz>();
        }
        __saga_core_function__ auto const &get_e() const {
          return this->template get<saga::property::e>();
        }
        __saga_core_function__ auto &get_e() {
          return this->template get<saga::property::e>();
        }
        __saga_core_function__ auto get_mass() const {
          return std::sqrt(std::abs(get_e() * get_e() - get_px() * get_px() -
                                    get_py() * get_py() - get_pz() * get_pz()));
        }
      };

      using value_type = value<particle_container>;
      using proxy_type = proxy<particle_container>;
      using const_proxy_type = const_proxy<particle_container>;
      using iterator_type = saga::core::proxy_iterator<particle_container>;
      using const_iterator_type =
          saga::core::const_proxy_iterator<particle_container>;

      using size_type = typename base_type::size_type;

      /// Access an element of the container
      __saga_core_function__ auto operator[](size_type idx) {
        return proxy_type(this, idx);
      }

      /// Access an element of the container
      __saga_core_function__ auto operator[](size_type idx) const {

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
