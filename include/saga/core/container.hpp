#pragma once
#include "saga/core/fields.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/types.hpp"

#include <tuple>
#include <type_traits>
#include <vector>

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

    static_assert(saga::types::is_valid_type_descriptor_v<TypeDescriptor>);

  public:
    using base_type = std::tuple<saga::core::container_t<
        saga::core::underlying_value_type_t<Field<TypeDescriptor>>,
        TypeDescriptor::backend>...>;

    using fields_type =
        saga::core::fields::fields_pack<Field<TypeDescriptor>...>;

    container_with_fields() = default;
    /// Construct the container with "n" elements
    container_with_fields(std::size_t n)
        : base_type(saga::core::container_t<
                    saga::core::underlying_value_type_t<Field<TypeDescriptor>>,
                    TypeDescriptor::backend>(n)...) {}
    container_with_fields(container_with_fields const &) = default;
    container_with_fields(container_with_fields &&) = default;
    container_with_fields &operator=(container_with_fields const &) = default;
    container_with_fields &operator=(container_with_fields &&) = default;

    /// Resize the container
    void resize(std::size_t n) {
      (std::get<saga::core::template_index_v<Field, Field...>>(*this).resize(n),
       ...);
    }

    /// Reserve elements
    void reserve(std::size_t n) {
      (std::get<saga::core::template_index_v<Field, Field...>>(*this).reserve(
           n),
       ...);
    }
    /// Number of elements
    std::size_t size() const { return std::get<0>(*this).size(); }

    // forward declarations
    class value_type;
    class proxy_type;
    class const_proxy_type;

    /* \brief A container value type
       This is not the actual type stored by the container, but rather a proxy
       to do operations with elements of a container.
     */
    class value_type
        : protected std::tuple<
              saga::core::underlying_value_type_t<Field<TypeDescriptor>>...> {

    public:
      using base_type = std::tuple<
          saga::core::underlying_value_type_t<Field<TypeDescriptor>>...>;

      value_type() = default;
      value_type(
          saga::core::underlying_value_type_t<Field<TypeDescriptor>> &&... v)
          : base_type(
                std::forward<
                    saga::core::underlying_value_type_t<Field<TypeDescriptor>>>(
                    v)...) {}
      value_type(
          saga::core::underlying_value_type_t<Field<TypeDescriptor>> const
              &... v)
          : base_type(v...) {}
      value_type(value_type const &) = default;
      value_type(value_type &&) = default;
      value_type &operator=(value_type &&) = default;
      value_type &operator=(value_type const &) = default;

      value_type(proxy_type const &p)
          : value_type(p.template get<Field>()...){};

      value_type &operator=(proxy_type const &p) {
        (set<Field>(p.template get<Field>()), ...);
        return *this;
      }
      value_type &operator=(const_proxy_type const &p) {
        (set<Field>(p.template get<Field>()), ...);
        return *this;
      }

      /// Get the value of the given field
      template <template <class> class F> auto const &get() const {
        return std::get<saga::core::template_index_v<F, Field...>>(*this);
      }

      /// Get the value of the given field
      template <template <class> class F> auto &get() {
        return std::get<saga::core::template_index_v<F, Field...>>(*this);
      }

      /// Set the values of all the fields
      template <template <class> class F>
      void set(saga::core::underlying_value_type_t<F<TypeDescriptor>> v) {
        std::get<saga::core::template_index_v<F, Field...>>(*this) = v;
      }
    };

    /* \brief A container proxy type
       This object is returned by containers when accessing a single element
     */
    class proxy_type {

    public:
      /// Container type
      using container_type = container_with_fields;

      /// Build the proxy from the container and the index
      proxy_type(container_type &cont, std::size_t idx)
          : m_ptr{&cont}, m_idx{idx} {}
      proxy_type(const proxy_type &other)
          : m_ptr{other.m_ptr}, m_idx{other.m_idx} {}
      proxy_type(proxy_type &&other) : m_ptr{other.m_ptr}, m_idx{other.m_idx} {}

      proxy_type &operator=(proxy_type const &other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }
      proxy_type &operator=(proxy_type &&other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }

      /// Assignment operator from a value type
      proxy_type &operator=(value_type const &other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }

      /// Assignment operator from a value type
      proxy_type &operator=(value_type &&other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }

      /// Set each element in the associated field of the container
      template <template <class> class F>
      void set(saga::core::underlying_value_type_t<F<TypeDescriptor>> v) {

        m_ptr->template set<F>(m_idx, v);
      }

      /// Pointer to the container
      container_type const *container_ptr() const { return m_ptr; }
      /// Container passed as a reference
      container_type &container() const { return *m_ptr; }
      /// Current index this proxy points to
      std::size_t index() const { return m_idx; }
      /// To allow STL-like operations
      proxy_type &operator*() { return *this; }
      /// To allow STL-like operations
      proxy_type const &operator*() const { return *this; }
      /// Increment the index
      proxy_type &operator++() {
        ++m_idx;
        return *this;
      }
      /// Increment the index
      proxy_type operator++(int) {

        auto copy = *this;
        ++(*this);
        return copy;
      }
      /// Decrement the index
      proxy_type &operator--() {
        --m_idx;
        return *this;
      }
      /// Decrement the index
      proxy_type operator--(int) {

        auto copy = *this;
        --(*this);
        return copy;
      }
      /// Get the value of one field from the container
      template <template <class> class F> auto const &get() const {
        return m_ptr->template get<F>()[m_idx];
      }
      /// Get the value of one field from the container
      template <template <class> class F> auto &get() {
        return m_ptr->template get<F>()[m_idx];
      }

    protected:
      /// Pointer to the container
      container_type *m_ptr = nullptr;
      /// Index in the container
      std::size_t m_idx = 0;
    };

    /* \brief A container proxy type
       This object is returned by containers when accessing a single element
     */
    class const_proxy_type {

    public:
      /// Container type
      using container_type = container_with_fields;

      /// Build the proxy from the container and the index
      const_proxy_type(container_type const &cont, std::size_t idx)
          : m_ptr{&cont}, m_idx{idx} {}
      const_proxy_type(const const_proxy_type &other)
          : m_ptr{other.m_ptr}, m_idx{other.m_idx} {}
      const_proxy_type(const_proxy_type &&other)
          : m_ptr{other.m_ptr}, m_idx{other.m_idx} {}

      /// Pointer to the container
      container_type const *container_ptr() const { return m_ptr; }
      /// Container as a reference
      container_type const &container() const { return *m_ptr; }
      /// Current index this proxy points to
      std::size_t index() const { return m_idx; }
      /// For compatibility with STL-like syntax
      const_proxy_type &operator*() { return *this; }
      /// For compatibility with STL-like syntax
      const_proxy_type &operator*() const { return *this; }
      /// Increment the index
      const_proxy_type &operator++() {
        ++m_idx;
        return *this;
      }
      /// Increment the index
      const_proxy_type operator++(int) {

        auto copy = *this;
        ++(*this);
        return copy;
      }
      /// Decrement the index
      const_proxy_type &operator--() {
        --m_idx;
        return *this;
      }
      /// Decrement the index
      const_proxy_type operator--(int) {

        auto copy = *this;
        --(*this);
        return copy;
      }
      /// Get the value of the field
      template <template <class> class F> auto const &get() const {
        return m_ptr->template get<F>()[m_idx];
      }

    protected:
      /// Pointer to the container
      container_type const *m_ptr = nullptr;
      /// Index in the container
      std::size_t m_idx = 0;
    };

    /// Comparison operator of two proxies
    friend bool operator==(proxy_type const &f, proxy_type const &s) {
      return (f.container_ptr() == s.container_ptr()) &&
             (f.index() == s.index());
    }
    /// Comparison operator of two proxies to constant containers
    friend bool operator==(const_proxy_type const &f,
                           const_proxy_type const &s) {
      return (f.container_ptr() == s.container_ptr()) &&
             (f.index() == s.index());
    }
    /// Comparison operator of two proxies with different cv qualifiers
    friend bool operator==(const_proxy_type const &f, proxy_type const &s) {
      return (f.container_ptr() == s.container_ptr()) &&
             (f.index() == s.index());
    }
    /// Comparison operator of two proxies with different cv qualifiers
    friend bool operator==(proxy_type const &f, const_proxy_type const &s) {
      return (f.container_ptr() == s.container_ptr()) &&
             (f.index() == s.index());
    }
    /// Comparison operator of two proxies
    friend bool operator!=(proxy_type const &f, proxy_type const &s) {
      return !(f == s);
    }
    /// Comparison operator of two proxies to constant containers
    friend bool operator!=(const_proxy_type const &f,
                           const_proxy_type const &s) {
      return !(f == s);
    }
    /// Comparison operator of two proxies with different cv qualifiers
    friend bool operator!=(const_proxy_type const &f, proxy_type const &s) {
      return !(f == s);
    }
    /// Comparison operator of two proxies with different cv qualifiers
    friend bool operator!=(proxy_type const &f, const_proxy_type const &s) {
      return !(f == s);
    }

    /// Access an element of the container
    auto operator[](std::size_t idx) { return proxy_type(*this, idx); }

    /// Access an element of the container
    auto operator[](std::size_t idx) const {

      return const_proxy_type(*this, idx);
    }

    /// Get the container associated to the given field
    template <template <class> class F> auto &get() {
      return std::get<saga::core::template_index_v<F, Field...>>(*this);
    }
    /// Get the container associated to the given field
    template <template <class> class F> auto const &get() const {
      return std::get<saga::core::template_index_v<F, Field...>>(*this);
    }
    /// Get the value associated to the given field and index in the container
    template <template <class> class F> auto &get(std::size_t i) {
      return this->template get<F>()[i];
    }
    /// Get the value associated to the given field and index in the container
    template <template <class> class F> auto const &get(std::size_t i) const {
      return this->template get<F>()[i];
    }
    /// Set the value associated to the given field and index in the container
    template <template <class> class F>
    void set(std::size_t i,
             saga::core::underlying_value_type_t<F<TypeDescriptor>> v) {
      this->template get<F>()[i] = v;
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
  };
} // namespace saga::core
