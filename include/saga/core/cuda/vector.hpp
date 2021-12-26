#pragma once
#include <stdexcept>
#include <string>
#include <types>

namespace saga::core::cuda {

  /// Represents the location of a GPU vector in memory
  enum memory_location { host, device };

  namespace detail {

    /// Base vector class, whose behaviour depends on the memory location
    template <class T, memory_location Location = host> class vector_ {

      static_assert(std::is_arithmetic_v<T>,
                    "A vector value type must be of arithmetic type");

      using value_type = T;
      using pointer_type = value_type *;
      using size_type = std::size_t;

      /// Create an empty vector
      vector_() = default;

      /// Free the memory used by the vector
      ~vector_() { clear(); }

      /// Construct a vector of the given size
      vector_(size_type n) { resize(n); }

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

        if constexpr (Location == host)
          m_data = new T[n];
        else {
          auto code = cudaMalloc(&m_data, n * sizeof(T));
          if (code != cudaSuccess) {
            auto megabytes = sizeof(T) * n / 1048576; // in MB
            throw std::runtime_error(
                std::string{"Unable to allocate vector of "} +
                std::to_string(megabytes) + " MB");
          }
        }
      }

      /// Free the memory used by the vector
      void clear() {
        if constexpr (Location == host) {
          if (m_data)
            delete[] m_data;
        } else {
          if (m_data) {
            auto code = cudaFree(m_data);
            if (code != cudaSuccess) {
              auto megabytes = sizeof(T) * m_size / 1048576;
              throw std::runtime_error(
                  std::string{"Problems freeing vector of size"} +
                  std::to_string(megabytes) + " MB");
            }
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
  } // namespace detail

  /// Vector class whose behaviour depends on the memory location
  template <class T, memory_location Location> class vector;

  /// Vector using the memory in the host
  template <class T> class vector<T, host> : public detail::vector_<T, host> {

  public:
    using base_class = detail::vector_<T, host>;
    using base_class::base_class;

    friend class detail::vector_<T, device>;

    /// Return a vector whose memory is allocated on the device
    auto to_device() const {

      vector<T, device> out;

      auto code = cudaMemcpy(out.m_data, m_data, m_size * sizeof(T),
                             cudaMemcpyHostToDevice);
      if (code != cudaSuccess)
        throw std::runtime_error("Unable to copy vector to the device");

      out.m_size = m_size;

      return out;
    }
  };

  /// Vector using the memory in the device
  template <class T>
  class vector<T, device> : public detail::vector_<T, device> {

  public:
    using base_class = detail::vector_<T, device>;
    using base_class::base_class;

    friend class detail::vector_<T, host>;

    /// Return a vector whose memory is allocated on the host
    vector<T, host> to_host() const {

      vector<T, host> out;

      auto code = cudaMemcpy(out.m_data, m_data, m_size * sizeof(T),
                             cudaMemcpyDeviceToHost);
      if (code != cudaSuccess)
        throw std::runtime_error("Unable to copy vector to host");

      out.m_size = m_size;

      return out;
    }
  };
} // namespace saga::core::cuda
