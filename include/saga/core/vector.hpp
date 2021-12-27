#pragma once
#include "saga/core/backend.hpp"

namespace saga {

  template<class T, saga::backend Backend>
  class vector;

  template<class T>
  class vector<T, saga::backend::CPU> {

  public:

    using value_type = T;
    using pointer_type = value_type*;
    using reference_type = value_type&;
    using const_reference_type = value_type const&;

    vector() = default;
    vector(size_type n) { resize(n); };
    vector(vector const& other) : m_data{allocate(other.m_size)}, m_size{other.m_size} {

      for ( auto i = 0u; i < other.m_size; ++i )
	this->operator[](i) = other[i];
    }
    vector(vector&& other) : m_data{other.m_data}, m_size{other.m_size} {

      other.m_data = nullptr;
      other.m_size = 0;
    }
    vector& operator=(vector const& other) {

      clear();

      m_data = allocate(other.m_size)
      m_size = other.m_size;

      for ( auto i = 0u; i < other.m_size; ++i )
	this->operator[](i) = other[i];

      return *this;
    }
    vector& operator=(vector&& other) {

      clear();

      m_data = other.m_data;
      m_size = other.m_size;

      other.m_data = nullptr;
      other.m_size = 0;
    }

    __saga_core_function__ const_reference_type operator[](size_type i) const { return m_data[i]; }

    __saga_core_function__ reference_type operator[](size_type i) { return m_data[i]; }

    __saga_core_function__ pointer_type data() const {
      return m_data;
    }

    void resize(size_type n) {

      clear();

      m_data = new value_type[n];
      m_size = n;
    }

    constexpr auto size() const {

      return m_size;
    }

  private:

    static pointer_type allocate(size_type n) {
      return n > 0 ? new value_type[n] : nullptr;
    }

    constexpr void clear() {

      if ( m_data ) {
	m_size = 0;
	delete[] m_data;
      }
    }

    pointer_type m_data = nullptr;
    size_type m_size = 0;
  };
}

#if SAGA_CUDA_ENABLED
#include "saga/core/cuda/vector.hpp"
#endif
