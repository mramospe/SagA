#pragma once
#include <stdexcept>
#include <string>
#include <types>
#include <vector>

#define SAGA_SIZE_IN_MEGABYTES 1048576

namespace saga::core::cuda {

  /// Vector class
  template <class T> class vector {

    static_assert(std::is_arithmetic_v<T>,
                  "A vector value type must be of arithmetic type");

    using value_type = T;
    using pointer_type = value_type *;
    using size_type = std::size_t;

    /// Create an empty vector
    vector() = default;

    /// Free the memory used by the vector
    ~vector() { clear(); }

    /// Construct a vector of the given size
    vector(size_type n) { resize(n); }

    /// Access the element at the given index
    __device__ __host__ T &operator[](size_type i) { return m_data[i]; }

    /// Access the element at the given index
    __device__ __host__ T const &operator[](size_type i) const {
      return m_data[i];
    }

    /// Resize the vector, clearing its contents beforehand
    void resize(size_type n) {

      clear();

      m_size = n;

      auto code = cudaMalloc(&m_data, n * sizeof(T));
      if (code != cudaSuccess) {
        auto megabytes = sizeof(T) * n / SAGA_SIZE_IN_MEGABYTES;
        throw std::runtime_error(std::string{"Unable to allocate vector of "} +
                                 std::to_string(megabytes) + " MB");
      }
    }

    /// Free the memory used by the vector
    void clear() {
      if (m_data) {
        auto code = cudaFree(m_data);
        if (code != cudaSuccess) {
          auto megabytes = sizeof(T) * m_size / SAGA_SIZE_IN_MEGABYTES;
          throw std::runtime_error(
              std::string{"Problems freeing vector of size"} +
              std::to_string(megabytes) + " MB");
        }
      }
    }

    /// Get the pointer of data
    auto data() const { return m_data; }

    /// Location of the array of data
    constexpr auto location() const { return Location; }

    /// Size of the vector
    __device__ __host__ size_type size() const { return m_size; }

  protected:
    /// The pointer to the memory
    pointer_type m_data = nullptr;
    /// Size of the vector
    size_type m_size = 0;
  };

  /// Return a vector whose memory is allocated on the device
  template <class T> auto to_device(std::vector<T> const &other) const {

    vector<T> out(other.size());

    auto code = cudaMemcpy(out.data(), other.data(), other.size() * sizeof(T),
                           cudaMemcpyHostToDevice);
    if (code != cudaSuccess)
      throw std::runtime_error("Unable to copy vector to the device");

    return out;
  }

  /// Return a vector whose memory is allocated on the host
  template <class T> auto to_host(vector<T> const &other) const {

    std::vector<T> out(other.size());

    auto code = cudaMemcpy(out.data(), other.data(), other.size() * sizeof(T),
                           cudaMemcpyDeviceToHost);
    if (code != cudaSuccess)
      throw std::runtime_error("Unable to copy vector to host");

    return out;
  }
} // namespace saga::core::cuda
