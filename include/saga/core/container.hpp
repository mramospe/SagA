#pragma once
#include "saga/core/iterator.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/proxy.hpp"
#include "saga/core/types.hpp"

#include <tuple>
#include <type_traits>

namespace saga::core {

  /*!\brief Standard container for objects with fields

   */
  template <class TypeDescriptor, template <class> class... Field>
  class container_with_fields
      : protected std::tuple<saga::core::container_t<
            saga::core::underlying_value_type_t<Field<TypeDescriptor>>,
            TypeDescriptor::backend>...> {

    static_assert(sizeof...(Field) > 0,
                  "Containers without fields make no sense");

    static_assert(saga::core::is_valid_type_descriptor_v<TypeDescriptor>,
                  "File descriptor is not valid");

  public:
    using type_descriptor = TypeDescriptor;

    using base_type = std::tuple<saga::core::container_t<
        saga::core::underlying_value_type_t<Field<type_descriptor>>,
        type_descriptor::backend>...>;

    using fields_type = saga::properties<Field...>;

    using size_type = std::size_t;

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
      (std::get<saga::core::template_index_v<Field, Field...>>(*this).resize(n),
       ...);
    }

    /// Reserve elements
    void reserve(size_type n) {
      (std::get<saga::core::template_index_v<Field, Field...>>(*this).reserve(
           n),
       ...);
    }
    /// Number of elements
    __saga_core_function__ size_type size() const {
      return std::get<0>(*this).size();
    }

    // forward declarations
    using value_type = saga::core::value<container_with_fields>;
    using proxy_type = saga::core::proxy<container_with_fields>;
    using const_proxy_type = saga::core::const_proxy<container_with_fields>;
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
      return std::get<saga::core::template_index_v<F, Field...>>(*this);
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
      return saga::core::is_template_in_v<Property, Field...>;
    }
    /// Set the value associated to the given field and index in the container
    template <template <class> class F>
    __saga_core_function__ void
    set(size_type i,
        saga::core::underlying_value_type_t<F<type_descriptor>> v) {
      std::get<saga::core::template_index_v<F, Field...>>(*this)[i] = v;
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
    void set(Container &&v) const {
      std::get<saga::core::template_index_v<F, Field...>>(*this) = v;
    }

    template <template <class> class F, class Container>
    void set(Container const &v) const {
      std::get<saga::core::template_index_v<F, Field...>>(*this) = v;
    }

#if SAGA_CUDA_ENABLED
  protected:
    /*\brief

     \warning: Meant to be used by inherited containers only
    */
    template <saga::backend NewBackend> auto to_backend() const {

      static_assert(NewBackend != type_descriptor::backend,
                    "Attempt to send data to the same backend");

      if constexpr (NewBackend == saga::backend::CUDA)
        return build_container_in_new_backend<NewBackend, Field...>(
            &saga::core::cuda::to_device);
      else
        return build_container_in_new_backend<NewBackend, Field...>(
            &saga::core::cuda::to_host);
    }

  private:
    /// Alias to a type where the backend has been switched to a one different
    /// from the previous
    template <saga::backend NewBackend>
    using type_with_backend =
        container_with_fields<saga::core::change_type_descriptor_backend_t<
                                  NewBackend, TypeDescriptor>,
                              Field...>;

    template <saga::backend NewBackend, class Function,
              template <class> class... F>
    auto build_container_in_new_backend(Function &&function) const {

      type_with_backend<NewBackend> cont;

      (cont.template set<F>(function(this->get<F>())), ...);

      return cont;
    }
#endif
  };
} // namespace saga::core
