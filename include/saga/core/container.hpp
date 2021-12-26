#pragma once
#include "saga/core/fields.hpp"
#include "saga/core/properties.hpp"
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

    using fields_type =
        saga::core::fields::fields_pack<Field<type_descriptor>...>;

    container_with_fields() = default;
    /// Construct the container with "n" elements
    container_with_fields(std::size_t n)
        : base_type(saga::core::container_t<
                    saga::core::underlying_value_type_t<Field<type_descriptor>>,
                    type_descriptor::backend>(n)...) {}
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
    class iterator_type;
    class proxy_type;
    class const_iterator_type;
    class const_proxy_type;

    /* \brief A container value type
       This is not the actual type stored by the container, but rather a proxy
       to do operations with elements of a container.
     */
    class value_type
        : protected std::tuple<
              saga::core::underlying_value_type_t<Field<type_descriptor>>...> {

    public:
      using base_type = std::tuple<
          saga::core::underlying_value_type_t<Field<type_descriptor>>...>;

      value_type() = default;
      value_type(
          saga::core::underlying_value_type_t<Field<type_descriptor>> &&...v)
          : base_type(std::forward<saga::core::underlying_value_type_t<
                          Field<type_descriptor>>>(v)...) {}
      value_type(
          saga::core::underlying_value_type_t<Field<type_descriptor>> const
              &...v)
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

      /// Whether this class has the specified property
      template <template <class> class Property> constexpr bool has() const {
        return saga::core::is_template_in_v<Property, Field...>;
      }

      /// Set the values of all the fields
      template <template <class> class F>
      void set(saga::core::underlying_value_type_t<F<type_descriptor>> v) {
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
      using container_pointer_type = container_type *;

      /// Build the proxy from the container and the index
      proxy_type(container_pointer_type cont, std::size_t idx)
          : m_ptr{cont}, m_idx{idx} {}
      /// The copy constructor assigns the internal container and index from the
      /// argument
      proxy_type(const proxy_type &) = default;
      /// The move constructor assigns the internal container and index from the
      /// argument
      proxy_type(proxy_type &&) = default;

      proxy_type &operator=(proxy_type const &other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }
      proxy_type &operator=(proxy_type &&other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }

      proxy_type &operator=(const_proxy_type const &other) {

        (set<Field>(other.template get<Field>()), ...);
        return *this;
      }
      proxy_type &operator=(const_proxy_type &&other) {

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
      void set(saga::core::underlying_value_type_t<F<type_descriptor>> v) {

        m_ptr->template set<F>(m_idx, v);
      }

      /// Whether this class has the specified property
      template <template <class> class Property> constexpr bool has() const {
        return saga::core::is_template_in_v<Property, Field...>;
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
      container_pointer_type container_ptr() { return m_ptr; }
      /// Container as a reference
      container_type &container() { return *m_ptr; }
      /// Container as a reference
      container_type const &container() const { return *m_ptr; }
      /// Current index this proxy points to
      std::size_t index() const { return m_idx; }

      /// Pointer to the container
      container_pointer_type m_ptr = nullptr;
      /// Index in the container
      std::size_t m_idx = 0;
    };

    /* \brief A container proxy type
       This object is returned by containers when accessing a single element
     */
    class iterator_type {

    public:
      /// Container type
      using container_type = container_with_fields;
      using container_pointer_type = container_type *;

      friend class const_iterator_type;

      /// Build the iterator from the container and the index
      iterator_type(container_type *cont, std::size_t idx)
          : m_ptr{cont}, m_idx{idx} {}
      iterator_type(const iterator_type &other) = default;
      iterator_type(iterator_type &&other) = default;
      iterator_type &operator=(iterator_type const &) = default;
      iterator_type &operator=(iterator_type &&) = default;

      /// To allow STL-like operations
      proxy_type operator*() { return proxy_type{m_ptr, m_idx}; }
      /// To allow STL-like operations
      const_proxy_type operator*() const { return proxy_type{m_ptr, m_idx}; }
      /// Increment the index
      iterator_type &operator++() {
        ++m_idx;
        return *this;
      }
      /// Increment the index
      iterator_type operator++(int) {

        auto copy = *this;
        ++(*this);
        return copy;
      }
      /// Decrement the index
      iterator_type &operator--() {
        --m_idx;
        return *this;
      }
      /// Decrement the index
      iterator_type operator--(int) {

        auto copy = *this;
        --(*this);
        return copy;
      }
      /// Move the internal index a certain quantity forward
      iterator_type &operator+=(int i) {
        m_idx += i;
        return *this;
      }
      /// Move the internal index a certain quantity backward
      iterator_type operator-=(int i) {
        m_idx -= i;
        return *this;
      }
      /// Return a copy of the iterator with the index modified by the given
      /// offset
      iterator_type operator+(int i) {
        auto copy = *this;
        copy += i;
        return copy;
      }
      /// Return a copy of the iterator with the index modified by the given
      /// offset
      iterator_type operator-(int i) {
        auto copy = *this;
        copy -= i;
        return copy;
      }
      /// Comparison operator of two proxies
      bool operator==(iterator_type const &s) {
        return (container_ptr() == s.container_ptr()) && (index() == s.index());
      }
      /// Comparison operator of two proxies with different cv qualifiers
      bool operator==(const_iterator_type const &s) {
        return (container_ptr() == s.container_ptr()) && (index() == s.index());
      }
      /// Comparison operator of two proxies
      bool operator!=(iterator_type const &s) { return !(*this == s); }
      /// Comparison operator of two proxies with different cv qualifiers
      bool operator!=(const_iterator_type const &s) { return !(*this == s); }

    protected:
      /// Pointer to the container
      container_pointer_type container_ptr() const { return m_ptr; }
      /// Container as a reference
      container_type const &container() const { return *m_ptr; }
      /// Current index this proxy points to
      std::size_t index() const { return m_idx; }

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
      using container_pointer_type = container_type const *;

      /// Build the proxy from the container and the index
      const_proxy_type(container_pointer_type cont, std::size_t idx)
          : m_ptr{cont}, m_idx{idx} {}
      /// The copy constructor assigns the internal container and index from the
      /// argument
      const_proxy_type(const const_proxy_type &) = default;
      /// The move constructor assigns the internal container and index from the
      /// argument
      const_proxy_type(const_proxy_type &&) = default;

      /// Whether this class has the specified property
      template <template <class> class Property> constexpr bool has() const {
        return saga::core::is_template_in_v<Property, Field...>;
      }
      /// Get the value of the field
      template <template <class> class F> auto const &get() const {
        return m_ptr->template get<F>()[m_idx];
      }

    protected:
      /// Pointer to the container
      container_pointer_type container_ptr() const { return m_ptr; }
      /// Container as a reference
      container_type const &container() const { return *m_ptr; }
      /// Current index this proxy points to
      std::size_t index() const { return m_idx; }

      /// Pointer to the container
      container_pointer_type m_ptr = nullptr;
      /// Index in the container
      std::size_t m_idx = 0;
    };

    /* \brief A container proxy type
       This object is returned by containers when accessing a single element
     */
    class const_iterator_type {

    public:
      /// Container type
      using container_type = container_with_fields;
      using container_pointer_type = container_type const *;

      friend class iterator_type;

      /// Build the proxy from the container and the index
      const_iterator_type(container_pointer_type cont, std::size_t idx)
          : m_ptr{cont}, m_idx{idx} {}
      const_iterator_type(const const_iterator_type &) = default;
      const_iterator_type(const_iterator_type &&) = default;
      const_iterator_type &operator=(const_iterator_type const &) = default;
      const_iterator_type &operator=(const_iterator_type &&) = default;

      /// For compatibility with STL-like syntax
      const_proxy_type operator*() { return const_proxy_type{m_ptr, m_idx}; }
      /// For compatibility with STL-like syntax
      const_proxy_type operator*() const {
        return const_proxy_type{m_ptr, m_idx};
      }
      /// Increment the index
      const_iterator_type &operator++() {
        ++m_idx;
        return *this;
      }
      /// Increment the index
      const_iterator_type operator++(int) {

        auto copy = *this;
        ++(*this);
        return copy;
      }
      /// Decrement the index
      const_iterator_type &operator--() {
        --m_idx;
        return *this;
      }
      /// Decrement the index
      const_iterator_type operator--(int) {

        auto copy = *this;
        --(*this);
        return copy;
      }
      /// Move the internal index a certain quantity forward
      const_iterator_type &operator+=(int i) {
        m_idx += i;
        return *this;
      }
      /// Move the internal index a certain quantity backward
      const_iterator_type operator-=(int i) {
        m_idx -= i;
        return *this;
      }
      /// Return a copy of the iterator with the index modified by the given
      /// offset
      const_iterator_type operator+(int i) {
        auto copy = *this;
        copy += i;
        return copy;
      }
      /// Return a copy of the iterator with the index modified by the given
      /// offset
      const_iterator_type operator-(int i) {
        auto copy = *this;
        copy -= i;
        return copy;
      }

      /// Comparison operator of two proxies to constant containers
      bool operator==(const_iterator_type const &s) {
        return (container_ptr() == s.container_ptr()) && (index() == s.index());
      }
      /// Comparison operator of two proxies with different cv qualifiers
      bool operator==(iterator_type const &s) {
        return (container_ptr() == s.container_ptr()) && (index() == s.index());
      }

      /// Comparison operator of two proxies to constant containers
      bool operator!=(const_iterator_type const &s) { return !(*this == s); }
      /// Comparison operator of two proxies with different cv qualifiers
      bool operator!=(iterator_type const &s) { return !(*this == s); }

    protected:
      /// Pointer to the container
      container_pointer_type container_ptr() const { return m_ptr; }
      /// Container as a reference
      container_type const &container() const { return *m_ptr; }
      /// Current index this proxy points to
      std::size_t index() const { return m_idx; }

      /// Pointer to the container
      container_pointer_type m_ptr = nullptr;
      /// Index in the container
      std::size_t m_idx = 0;
    };

    /// Access an element of the container
    auto operator[](std::size_t idx) { return proxy_type(this, idx); }

    /// Access an element of the container
    auto operator[](std::size_t idx) const {

      return const_proxy_type(this, idx);
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
    /// Whether this class has the specified property
    template <template <class> class Property> constexpr bool has() const {
      return saga::core::is_template_in_v<Property, Field...>;
    }
    /// Set the value associated to the given field and index in the container
    template <template <class> class F>
    void set(std::size_t i,
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
    auto to_device() const {
      return build_container_in_new_backend<saga::backend::CUDA, Field...>(
          &saga::core::cuda::to_device);
    }

    /*\brief

      \warning: Meant to be used by inherited containers only
    */
    auto to_host() const {
      return build_container_in_new_backend<saga::backend::CPU, Field...>(
          &saga::core::cuda::to_host);
    }

  private:
    /// Alias to a type where the backend has been switched to a one different
    /// from the previous
    template <saga::backend NewBackend>
    using type_with_backend =
        container_with_fields<saga::core::switch_type_descriptor_backend_t<
                                  NewBackend, TypeDescriptor>,
                              Field...>;

    template <saga::backend NewBackend, class Function,
              template <class> class... Field>
    auto build_container_in_new_backend(Function &&function) const {

      type_with_backend<NewBackend> cont;

      (cont.set<Field>(function(this->get<Field>()))...);

      return cont;
    }
#endif
  };
} // namespace saga::core
