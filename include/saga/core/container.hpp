#pragma once
#include "saga/core/fields.hpp"
#include "saga/core/properties.hpp"
#include "saga/core/types.hpp"

#include <vector>

namespace saga::core {

  /// Reserve elements on a STL vector
  template <class TypeDescriptor> struct container_reserve {
    template <class T>
    constexpr std::enable_if_t<
        types::is_type_descriptor_type_v<TypeDescriptor, std::decay_t<T>>, void>
    operator()(std::vector<T> &v, std::size_t n) const {
      v.reserve(n);
    }
  };

  /// Change the size of a STL vector
  template <class TypeDescriptor> struct container_resize {
    template <class T>
    constexpr std::enable_if_t<
        types::is_type_descriptor_type_v<TypeDescriptor, std::decay_t<T>>, void>
    operator()(std::vector<T> &v, std::size_t n) const {
      v.resize(n);
    }
  };

  /// Size of an STL vector
  template <class TypeDescriptor> struct container_size {
    template <class T>
    constexpr std::enable_if_t<
        types::is_type_descriptor_type_v<TypeDescriptor, std::decay_t<T>>,
        typename std::vector<T>::size_type>
    operator()(std::vector<T> const &v) const {
      return v.size();
    }
  };

  /*!\brief Standard container for objects with fields

   */
  template <class TypeDescriptor, template <class> class... Field>
  class container_with_fields
      : public Field<TypeDescriptor>::container_type... {

    static_assert(sizeof...(Field) > 0,
                  "Containers without fields make no sense");

    static_assert(saga::types::is_valid_type_descriptor_v<TypeDescriptor>);

  public:
    using fields_type =
        saga::core::fields::fields_pack<Field<TypeDescriptor>...>;

    container_with_fields() = default;
    /// Construct the container with "n" elements
    container_with_fields(std::size_t n)
        : Field<TypeDescriptor>::container_type{n}... {}
    container_with_fields(container_with_fields const &) = default;
    container_with_fields(container_with_fields &&) = default;
    container_with_fields &operator=(container_with_fields const &) = default;
    container_with_fields &operator=(container_with_fields &&) = default;

    /// Resize the container
    void resize(std::size_t n) {
      fields::visitor<fields_type>::invoke_void(
          saga::core::container_resize<TypeDescriptor>{}, *this, n);
    }

    /// Reserve elements
    void reserve(std::size_t n) {
      fields::visitor<fields_type>::invoke_void(
          saga::core::container_reserve<TypeDescriptor>{}, *this, n);
    }
    /// Number of elements
    std::size_t size() const {
      return fields::visitor<fields_type>::invoke_first(
          saga::core::container_size<TypeDescriptor>{}, *this);
    }

    /* \brief A container value type
       This is not the actual type stored by the container, but rather a proxy
       to do operations with elements of a container.
     */
    class value_type : public Field<TypeDescriptor>::value_type... {

    public:
      value_type() = default;
      value_type(typename Field<TypeDescriptor>::underlying_value_type &&... v)
          : Field<TypeDescriptor>::value_type(
                std::forward<
                    typename Field<TypeDescriptor>::underlying_value_type>(
                    v))... {}
      value_type(
          typename Field<TypeDescriptor>::underlying_value_type const &... v)
          : Field<TypeDescriptor>::value_type(v)... {}
      value_type(value_type const &) = default;
      value_type(value_type &&) = default;
      value_type &operator=(value_type &&) = default;
      value_type &operator=(value_type const &) = default;

      /// Get the value of the given field
      template <template <class> class F> auto const &get() const {
        return F<TypeDescriptor>::value_type::get(*this);
      }

      /// Get the value of the given field
      template <template <class> class F> auto &get() {
        return F<TypeDescriptor>::value_type::get(*this);
      }

      /// Set the values of all the fields
      void set(typename Field<TypeDescriptor>::underlying_value_type... v) {
        (Field<TypeDescriptor>::value_type::set(*this, v), ...);
      }
    };

    /* \brief A container proxy type
       This object is returned by containers when accessing a single element
     */
    class proxy_type : public Field<TypeDescriptor>::template proxy_type<
                           container_with_fields>... {
    public:
      /// Container type
      using container_type = container_with_fields;

      /// Build the proxy from the container and the index
      proxy_type(container_type &cont, std::size_t idx)
          : m_ptr{&cont}, m_idx{idx} {}

      /// Assignment operator from a value type
      proxy_type &operator=(value_type const &other) {

        (Field<TypeDescriptor>::template proxy_type<container_type>::set(
             *this, Field<TypeDescriptor>::value_type::get(other)),
         ...);

        return *this;
      }

      /// Set each element in the associated field of the container
      void set(typename Field<TypeDescriptor>::underlying_value_type... value) {

        (Field<TypeDescriptor>::template proxy_type<container_type>::set(*this,
                                                                         value),
         ...);
      }

      /// Pointer to the container
      container_type const *container_ptr() const { return m_ptr; }
      /// Container passed as a reference
      container_type &container() const override { return *m_ptr; }
      /// Current index this proxy points to
      std::size_t index() const override { return m_idx; }
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
        return F<TypeDescriptor>::template proxy_type<container_type>::get(
            *this);
      }
      /// Get the value of one field from the container
      template <template <class> class F> auto &get() {
        return F<TypeDescriptor>::template proxy_type<container_type>::get(
            *this);
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
    class const_proxy_type
        : public Field<TypeDescriptor>::template const_proxy_type<
              container_with_fields>... {
    public:
      /// Container type
      using container_type = container_with_fields;

      /// Build the proxy from the container and the index
      const_proxy_type(container_type const &cont, std::size_t idx)
          : m_ptr{&cont}, m_idx{idx} {}

      /// Pointer to the container
      container_type const *container_ptr() const { return m_ptr; }
      /// Container as a reference
      container_type const &container() const override { return *m_ptr; }
      /// Current index this proxy points to
      std::size_t index() const override { return m_idx; }
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
        return F<TypeDescriptor>::template const_proxy_type<
            container_type>::get(*this);
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
      return F<TypeDescriptor>::container_type::get(*this);
    }
    /// Get the container associated to the given field
    template <template <class> class F> auto const &get() const {
      return F<TypeDescriptor>::container_type::get(*this);
    }
    /// Get the value associated to the given field and index in the container
    template <template <class> class F> auto &get(std::size_t i) {
      return F<TypeDescriptor>::container_type::get(*this, i);
    }
    /// Get the value associated to the given field and index in the container
    template <template <class> class F> auto const &get(std::size_t i) const {
      return F<TypeDescriptor>::container_type::get(*this, i);
    }
    /// Set the value associated to the given field and index in the container
    template <template <class> class F>
    void set(std::size_t i,
             typename F<TypeDescriptor>::underlying_value_type v) {
      F<TypeDescriptor>::container_type::set(*this, i, v);
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
           saga::core::template_at<I, Field...>::template tpl>(
           std::forward<ElementType>(el)),
       ...);
    }

    template <template <class> class F>
    void push_back_impl_single(value_type const &el) {
      F<TypeDescriptor>::container_type::get(*this).push_back(
          F<TypeDescriptor>::value_type::get(el));
    }

    template <template <class> class F>
    void push_back_impl_single(value_type &&el) {
      F<TypeDescriptor>::container_type::get(*this).push_back(
          std::forward<value_type::value_type>(
              F<TypeDescriptor>::value_type::get(el)));
    }

    template <template <class> class F>
    void push_back_impl_single(proxy_type const &el) {
      using aux_proxy_type = typename F<TypeDescriptor>::template proxy_type<
          container_with_fields>;
      F<TypeDescriptor>::container_type::get(*this).push_back(
          aux_proxy_type::get(el));
    }

    template <template <class> class F>
    void push_back_impl_single(proxy_type &&el) {
      using aux_proxy_type = typename F<TypeDescriptor>::template proxy_type<
          container_with_fields>;
      F<TypeDescriptor>::container_type::get(*this).push_back(
          std::forward<proxy_type::proxy_type>(aux_proxy_type::get(el)));
    }

    template <template <class> class F>
    void push_back_impl_single(const_proxy_type const &el) {
      using aux_const_proxy_type = typename F<
          TypeDescriptor>::template const_proxy_type<container_with_fields>;
      F<TypeDescriptor>::container_type::get(*this).push_back(
          aux_const_proxy_type::get(el));
    }
  };
} // namespace saga::core
