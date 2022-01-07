#pragma once
#include "saga/cuda/core.hpp"
#include <limits>
#include <tuple>

namespace saga::cuda {

  /// Set forces to zero
  template <class View, class ValueType>
  __global__ void set_view_values(View obj, ValueType def) {

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid < obj.size())
      obj[gtid] = def;
  }

  namespace detail {
    // workaround to allow calling "std::numeric_limits::max" as a constant
    // expression inside a kernel
    template <class TypeDescriptor> struct numeric_info {
      static constexpr auto float_max =
          std::numeric_limits<typename TypeDescriptor::float_type>::max();
    };
  } // namespace detail

  /*!\brief Find the combination for which *predicate* is the smallest

    The predicate is assumed to return both the value and the result of a check
    for its validation. If no suitable combination is found then the returned
    index is -1.
   */
  template <class IndicesView, class View, class Functor, class... Args>
  __global__ void find_lesser_with_validation(IndicesView out, View obj,
                                              Functor predicate, Args... args) {

    extern __shared__ char shared_memory[];

    // CUDA does not allow to create arrays of template types yet
    auto shared_objects = (typename View::value_type *)shared_memory;

    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    auto current = gtid < obj.size() ? obj[gtid] : obj[0];

    // the number of tiles is equal to the dimension of the block
    auto ntiles = obj.size() / blockDim.x + (obj.size() % blockDim.x != 0);

    auto result =
        detail::numeric_info<typename View::type_descriptor>::float_max;

    typename IndicesView::value_type index = gtid;
    bool is_valid = false;

    // loop over the tiles
    for (auto tile = 0u; tile < ntiles; ++tile) {

      auto idx = tile * blockDim.x + threadIdx.x;

      if (idx < obj.size())
        shared_objects[threadIdx.x] = obj[idx];

      // the shared memory has been filled
      __syncthreads();

      if (gtid < obj.size()) {
        for (auto i = 0u; i < blockDim.x; ++i) {

          auto cid = tile * blockDim.x + i;

          if (cid == gtid)
            continue;
          else if (cid < obj.size()) {

            auto [new_result, new_valid] =
                predicate(current, shared_objects[i], args...);

            if (new_valid && new_result < result) {
              result = new_result;
              is_valid = new_valid;
              index = cid;
            }

          } else
            break;
        }
      }

      // the interactions have been computed
      __syncthreads();
    }

    if (gtid < obj.size())
      out[gtid] = is_valid ? index : SAGA_CUDA_INVALID_INDEX;
  }

  /// Apply a functor on the given view
  template <class View, class Functor, class... Args>
  __global__ void apply_functor(View obj, Functor functor, Args... args) {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < obj.size())
      functor(obj[gtid], args...);
  }

  /// Apply a functor using the information from the two views
  template <class FirstView, class SecondView, class Functor, class... Args>
  __global__ void
  apply_functor_contiguous_views(FirstView first, SecondView second,
                                 Functor functor, Args... args) {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < first.size())
      functor(first[gtid], second[gtid], args...);
  }
} // namespace saga::cuda
