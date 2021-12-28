#pragma once
#include "saga/core/backend.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>

#define SAGA_SIZE_IN_MEGABYTES 1048576

#define SAGA_THROW_CUDA_ERROR                                                  \
  throw std::runtime_error("Attempt to run with CUDA backend on software "     \
                           "compiled exclusively for CPU")

namespace saga {

  template <class T, saga::backend Backend> class vector {

    static_assert(std::is_arithmetic_v<T>,
                  "A vector value type must be of arithmetic type");

#if not SAGA_CUDA_ENABLED
    static_assert(Backend == saga::backend::CPU,
                  "Can not initialize vector for the CUDA backend if the "
                  "compilation is done for CPU only");
#endif

  public:
    using value_type = T;
    using pointer_type = value_type *;
    using reference_type = value_type &;
    using const_reference_type = value_type const &;
    using size_type = std::size_t;

    __saga_core_function__ vector() = default;
    __saga_core_function__ vector(size_type n) { resize(n); };
    __saga_core_function__ vector(vector const &other)
        : m_data{allocate(other.m_size)}, m_size{other.m_size} {

      for (auto i = 0u; i < other.m_size; ++i)
        this->operator[](i) = other[i];
    }
    __saga_core_function__ vector(vector &&other)
        : m_data{other.m_data}, m_size{other.m_size} {

      other.m_data = nullptr;
      other.m_size = 0;
    }
    __saga_core_function__ vector &operator=(vector const &other) {

      clear();

      m_data = allocate(other.m_size);
      m_size = other.m_size;

      for (auto i = 0u; i < other.m_size; ++i)
        this->operator[](i) = other[i];

      return *this;
    }
    __saga_core_function__ vector &operator=(vector &&other) {

      clear();

      m_data = other.m_data;
      m_size = other.m_size;

      other.m_data = nullptr;
      other.m_size = 0;
    }

    __saga_core_function__ const_reference_type operator[](size_type i) const {
      return m_data[i];
    }

    __saga_core_function__ reference_type operator[](size_type i) {
      return m_data[i];
    }

    __saga_core_function__ pointer_type data() const { return m_data; }

    void resize(size_type n) {

      clear();

      m_data = allocate(n);
      m_size = n;
    }

    constexpr auto size() const { return m_size; }

    void clear() {

      if constexpr (Backend == saga::backend::CPU) {
        if (m_data) {
          m_size = 0;
          delete[] m_data;
        }
      } else {
#if SAGA_CUDA_ENABLED
        if (m_data) {
          auto code = cudaFree(m_data);
          if (code != cudaSuccess) {
            auto megabytes = sizeof(T) * m_size / SAGA_SIZE_IN_MEGABYTES;
            throw std::runtime_error(
                std::string{"Problems freeing vector of size"} +
                std::to_string(megabytes) + " MB");
          }
        }
#else
        SAGA_THROW_CUDA_ERROR;
#endif
      }
    }

  private:
    static pointer_type allocate(size_type n) {

      if constexpr (Backend == saga::backend::CPU)
        return n > 0 ? new value_type[n] : nullptr;
      else {
#if SAGA_CUDA_ENABLED
        auto code = cudaMalloc(&m_data, n * sizeof(T));
        if (code != cudaSuccess) {
          auto megabytes = sizeof(T) * n / SAGA_SIZE_IN_MEGABYTES;
          throw std::runtime_error(
              std::string{"Unable to allocate vector of "} +
              std::to_string(megabytes) + " MB");
        }
#else
        SAGA_THROW_CUDA_ERROR;
#endif
      }
    }

    pointer_type m_data = nullptr;
    size_type m_size = 0;
  };

#if SAGA_CUDA_ENABLED
  /// Return a vector whose memory is allocated on the device
  template <class T>
  auto to_device(vector<T, saga::backend::CPU> const &other) {

    vector<T, saga::CUDA> out(other.size());

    auto code = cudaMemcpy(out.data(), other.data(), other.size() * sizeof(T),
                           cudaMemcpyHostToDevice);
    if (code != cudaSuccess)
      throw std::runtime_error("Unable to copy vector to the device");

    return out;
  }

  /// Return a vector whose memory is allocated on the host
  template <class T> auto to_host(vector<T, saga::backend::CUDA> const &other) {

    vector<T, saga::CPU> out(other.size());

    auto code = cudaMemcpy(out.data(), other.data(), other.size() * sizeof(T),
                           cudaMemcpyDeviceToHost);
    if (code != cudaSuccess)
      throw std::runtime_error("Unable to copy vector to host");

    return out;
  }
#endif
} // namespace saga
