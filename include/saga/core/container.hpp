#pragma once
#include "saga/core/backend.hpp"
#include "saga/core/iterator.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/proxy.hpp"
#include "saga/core/tuple.hpp"
#include "saga/core/vector.hpp"
#include "saga/core/views.hpp"

#include <type_traits>

#if SAGA_CUDA_ENABLED
namespace saga {

  namespace core::detail {

    template <saga::backend NewBackend> struct to_backend_t;

    template <> struct to_backend_t<saga::backend::CPU> {

      template <class Container, template <class> class... Field>
      auto operator()(Container const &container,
                      saga::properties<Field...>) const {

        using type_descriptor = typename Container::type_descriptor;

        static_assert((type_descriptor::backend != saga::backend::CPU),
                      "Sending a container to the same backend");

        typename Container::template type_with_backend<saga::backend::CPU> out;

        (out.template set<Field>(
             saga::core::to_host(container.template get<Field>())),
         ...);

        return out;
      }
    };

    template <> struct to_backend_t<saga::backend::CUDA> {

      template <class Container, template <class> class... Field>
      auto operator()(Container const &container,
                      saga::properties<Field...>) const {

        using type_descriptor = typename Container::type_descriptor;

        static_assert((type_descriptor::backend != saga::backend::CUDA),
                      "Sending a container to the same backend");

        typename Container::template type_with_backend<saga::backend::CUDA> out;

        (out.template set<Field>(
             saga::core::to_device(container.template get<Field>())),
         ...);

        return out;
      }
    };

  } // namespace core::detail

  template <saga::backend NewBackend, class Container>
  auto to_backend(Container const &container) {

    return saga::core::detail::to_backend_t<NewBackend>{}(
        container, typename Container::fields_type{});
  }
} // namespace saga
#endif

namespace saga::core {

  /// Container for the given backend
  template <class T, saga::backend Backend> struct container;

  /// Container for the CPU backend
  template <class T> struct container<T, saga::backend::CPU> {
    using type = saga::core::vector<T, saga::backend::CPU>;
  };

#if SAGA_CUDA_ENABLED
  /// Container for the CUDA backend
  template <class T> struct container<T, saga::backend::CUDA> {
    using type = saga::core::vector<T, saga::backend::CUDA>;
  };
#endif

  /// Alias to get the type of a container for a given backend
  template <class T, backend Backend>
  using container_t = typename container<T, Backend>::type;

  /*!\brief Standard container for objects with fields

   */
  template <class TypeDescriptor, template <class> class... Field>
  class container_with_fields
      : protected saga::core::tuple<saga::core::container_t<
            saga::core::underlying_value_type_t<Field<TypeDescriptor>>,
            TypeDescriptor::backend>...> {

    template <class Container, template <class> class... F>
    friend auto
    saga::core::detail::make_vector_views_impl(Container &,
                                               saga::properties<F...>);

#if SAGA_CUDA_ENABLED
    template <saga::backend> friend class saga::core::detail::to_backend_t;
#endif

    static_assert(sizeof...(Field) > 0,
                  "Containers without fields make no sense");

    static_assert(saga::core::is_valid_type_descriptor_v<TypeDescriptor>,
                  "File descriptor is not valid");

  public:
    using type_descriptor = TypeDescriptor;

    using base_type = saga::core::tuple<saga::core::container_t<
        saga::core::underlying_value_type_t<Field<type_descriptor>>,
        type_descriptor::backend>...>;

    using fields_type = saga::properties<Field...>;

    using size_type = std::size_t;

#if SAGA_CUDA_ENABLED
    /// Alias to a type where the backend has been switched to a one different
    /// from the previous
    template <saga::backend NewBackend>
    using type_with_backend =
        container_with_fields<saga::core::change_type_descriptor_backend_t<
                                  NewBackend, type_descriptor>,
                              Field...>;
#endif

    container_with_fields() = default;
    /// Construct the container with "n" elements
    container_with_fields(size_type n)
        : base_type(saga::core::container_t<
                    saga::core::underlying_value_type_t<Field<type_descriptor>>,
                    type_descriptor::backend>(n)...) {}
    container_with_fields(container_with_fields const &) = default;
    container_with_fields(container_with_fields &&) = default;
    container_with_fields &operator=(container_with_fields const &) = default;
    container_with_fields &operator=(container_with_fields &&) = default;

    /// Resize the container
    void resize(size_type n) {
      (saga::core::get<saga::core::template_index_v<Field, Field...>>(*this)
           .resize(n),
       ...);
    }

    /// Number of elements
    __saga_core_function__ size_type size() const {
      return saga::core::get<0>(*this).size();
    }

    // forward declarations
    template <class ContainerOrView>
    using value = saga::core::value<ContainerOrView>;

    template <class ContainerOrView>
    using proxy = saga::core::proxy<ContainerOrView>;

    template <class ContainerOrView>
    using const_proxy = saga::core::const_proxy<ContainerOrView>;

    using value_type = value<container_with_fields>;
    using proxy_type = proxy<container_with_fields>;
    using const_proxy_type = const_proxy<container_with_fields>;
    using iterator_type = saga::core::proxy_iterator<container_with_fields>;
    using const_iterator_type =
        saga::core::const_proxy_iterator<container_with_fields>;

    /// Access an element of the container
    __saga_core_function__ auto operator[](size_type idx) {
      return proxy_type(this, idx);
    }

    /// Access an element of the container
    __saga_core_function__ auto operator[](size_type idx) const {

      return const_proxy_type(this, idx);
    }

    /// Get the container associated to the given field
    template <template <class> class F>
    __saga_core_function__ auto const &get() const {
      return saga::core::get<saga::core::template_index_v<F, Field...>>(*this);
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
    /// Whether this class has the specified property
    template <template <class> class Property>
    constexpr __saga_core_function__ bool has() const {
      return saga::core::has_single_template_v<Property, Field...>;
    }
    /// Set the value associated to the given field and index in the container
    template <template <class> class F>
    __saga_core_function__ void
    set(size_type i,
        saga::core::underlying_value_type_t<F<type_descriptor>> v) {
      saga::core::get<saga::core::template_index_v<F, Field...>>(*this)[i] = v;
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

    /// Add a new element to the container
    void push_back(value_type const &el) {
      push_back_impl(el, std::make_index_sequence<sizeof...(Field)>());
    }

    /// Add a new element to the container
    void push_back(value_type &&el) {
      push_back_impl(el, std::make_index_sequence<sizeof...(Field)>());
    }

    /// Add a new element to the container
    void push_back(proxy_type const &el) {
      push_back_impl(el, std::make_index_sequence<sizeof...(Field)>());
    }

    /// Add a new element to the container
    void push_back(proxy_type &&el) {
      push_back_impl(el, std::make_index_sequence<sizeof...(Field)>());
    }

    /// Add a new element to the container
    void push_back(const_proxy_type const &el) {
      push_back_impl(el, std::make_index_sequence<sizeof...(Field)>());
    }

  private:
    template <class ElementType, std::size_t... I>
    void push_back_impl(ElementType const &el, std::index_sequence<I...>) {
      (push_back_impl_single<
           saga::core::template_at<I, Field...>::template tpl>(el),
       ...);
    }

    template <class ElementType, std::size_t... I>
    void push_back_impl(ElementType &&el, std::index_sequence<I...>) {
      (push_back_impl_single<
           saga::core::template_at<I, Field...>::template tpl>(el),
       ...);
    }

    template <template <class> class F, class ElementType>
    void push_back_impl_single(ElementType const &el) {
      this->template get<F>().push_back(el.template get<F>());
    }

    /// Set the container associated to the given field
    template <template <class> class F, class Container>
    void set(Container &&v) {
      saga::core::get<saga::core::template_index_v<F, Field...>>(*this) =
          std::forward<Container>(v);
    }

    template <template <class> class F, class Container>
    void set(Container const &v) {
      saga::core::get<saga::core::template_index_v<F, Field...>>(*this) = v;
    }

    /// Get the container associated to the given field
    template <template <class> class F>
    __saga_core_function__ auto &get_non_const() {
      return saga::core::get<saga::core::template_index_v<F, Field...>>(*this);
    }
  };
} // namespace saga::core
