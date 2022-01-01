#pragma once
#include <cstdlib>

namespace saga::core {

  // forward declarations
  template <class> class proxy_iterator;
  template <class> class const_proxy_iterator;

  /* \brief A container proxy type
     This object is returned by containers when accessing a single element
  */
  template <class Container> class proxy_iterator {

  public:
    /// Container type
    using container_type = Container;
    using container_pointer_type = container_type *;
    using proxy_type = typename container_type::proxy_type;
    using const_proxy_type = typename container_type::const_proxy_type;
    using size_type = std::size_t;
    using const_iterator_type = const_proxy_iterator<container_type>;

    friend class const_proxy_iterator<container_type>;

    proxy_iterator() = delete;
    /// Build the iterator from the container and the index
    proxy_iterator(container_type *cont, size_type idx)
        : m_ptr{cont}, m_idx{idx} {}
    proxy_iterator(const proxy_iterator &other) = default;
    proxy_iterator(proxy_iterator &&other) = default;
    proxy_iterator &operator=(proxy_iterator const &) = default;
    proxy_iterator &operator=(proxy_iterator &&) = default;

    /// To allow STL-like operations
    proxy_type operator*() { return proxy_type{m_ptr, m_idx}; }
    /// To allow STL-like operations
    const_proxy_type operator*() const { return proxy_type{m_ptr, m_idx}; }
    /// Increment the index
    proxy_iterator &operator++() {
      ++m_idx;
      return *this;
    }
    /// Increment the index
    proxy_iterator operator++(int) {

      auto copy = *this;
      ++(*this);
      return copy;
    }
    /// Decrement the index
    proxy_iterator &operator--() {
      --m_idx;
      return *this;
    }
    /// Decrement the index
    proxy_iterator operator--(int) {

      auto copy = *this;
      --(*this);
      return copy;
    }
    /// Move the internal index a certain quantity forward
    proxy_iterator &operator+=(int i) {
      m_idx += i;
      return *this;
    }
    /// Move the internal index a certain quantity backward
    proxy_iterator operator-=(int i) {
      m_idx -= i;
      return *this;
    }
    /// Return a copy of the iterator with the index modified by the given
    /// offset
    proxy_iterator operator+(int i) {
      auto copy = *this;
      copy += i;
      return copy;
    }
    /// Return a copy of the iterator with the index modified by the given
    /// offset
    proxy_iterator operator-(int i) {
      auto copy = *this;
      copy -= i;
      return copy;
    }
    /// Comparison operator of two proxies
    bool operator==(proxy_iterator const &s) {
      return (container_ptr() == s.container_ptr()) && (index() == s.index());
    }
    /// Comparison operator of two proxies with different cv qualifiers
    bool operator==(const_iterator_type const &s) {
      return (container_ptr() == s.container_ptr()) && (index() == s.index());
    }
    /// Comparison operator of two proxies
    bool operator!=(proxy_iterator const &s) { return !(*this == s); }
    /// Comparison operator of two proxies with different cv qualifiers
    bool operator!=(const_iterator_type const &s) { return !(*this == s); }

  protected:
    /// Pointer to the container
    container_pointer_type container_ptr() const { return m_ptr; }
    /// Container as a reference
    container_type const &container() const { return *m_ptr; }
    /// Current index this proxy points to
    constexpr size_type index() const { return m_idx; }

    /// Pointer to the container
    container_pointer_type m_ptr = nullptr;
    /// Index in the container
    size_type m_idx = 0;
  };

  /* \brief A container proxy type
     This object is returned by containers when accessing a single element
  */
  template <class Container> class const_proxy_iterator {

  public:
    /// Container type
    using container_type = Container;
    using container_pointer_type = container_type const *;
    using proxy_type = typename container_type::proxy_type;
    using const_proxy_type = typename container_type::const_proxy_type;
    using size_type = std::size_t;
    using iterator_type = proxy_iterator<container_type>;

    friend class proxy_iterator<container_type>;

    const_proxy_iterator() = delete;
    /// Build the proxy from the container and the index
    const_proxy_iterator(container_pointer_type cont, size_type idx)
        : m_ptr{cont}, m_idx{idx} {}
    const_proxy_iterator(const const_proxy_iterator &) = default;
    const_proxy_iterator(const_proxy_iterator &&) = default;
    const_proxy_iterator &operator=(const_proxy_iterator const &) = default;
    const_proxy_iterator &operator=(const_proxy_iterator &&) = default;

    /// For compatibility with STL-like syntax
    const_proxy_type operator*() { return const_proxy_type{m_ptr, m_idx}; }
    /// For compatibility with STL-like syntax
    const_proxy_type operator*() const {
      return const_proxy_type{m_ptr, m_idx};
    }
    /// Increment the index
    const_proxy_iterator &operator++() {
      ++m_idx;
      return *this;
    }
    /// Increment the index
    const_proxy_iterator operator++(int) {

      auto copy = *this;
      ++(*this);
      return copy;
    }
    /// Decrement the index
    const_proxy_iterator &operator--() {
      --m_idx;
      return *this;
    }
    /// Decrement the index
    const_proxy_iterator operator--(int) {

      auto copy = *this;
      --(*this);
      return copy;
    }
    /// Move the internal index a certain quantity forward
    const_proxy_iterator &operator+=(int i) {
      m_idx += i;
      return *this;
    }
    /// Move the internal index a certain quantity backward
    const_proxy_iterator operator-=(int i) {
      m_idx -= i;
      return *this;
    }
    /// Return a copy of the iterator with the index modified by the given
    /// offset
    const_proxy_iterator operator+(int i) {
      auto copy = *this;
      copy += i;
      return copy;
    }
    /// Return a copy of the iterator with the index modified by the given
    /// offset
    const_proxy_iterator operator-(int i) {
      auto copy = *this;
      copy -= i;
      return copy;
    }

    /// Comparison operator of two proxies to constant containers
    bool operator==(const_proxy_iterator const &s) {
      return (container_ptr() == s.container_ptr()) && (index() == s.index());
    }
    /// Comparison operator of two proxies with different cv qualifiers
    bool operator==(iterator_type const &s) {
      return (container_ptr() == s.container_ptr()) && (index() == s.index());
    }

    /// Comparison operator of two proxies to constant containers
    bool operator!=(const_proxy_iterator const &s) { return !(*this == s); }
    /// Comparison operator of two proxies with different cv qualifiers
    bool operator!=(iterator_type const &s) { return !(*this == s); }

  protected:
    /// Pointer to the container
    container_pointer_type container_ptr() const { return m_ptr; }
    /// Container as a reference
    container_type const &container() const { return *m_ptr; }
    /// Current index this proxy points to
    constexpr size_type index() const { return m_idx; }

    /// Pointer to the container
    container_pointer_type m_ptr = nullptr;
    /// Index in the container
    size_type m_idx = 0;
  };
} // namespace saga::core
