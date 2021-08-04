#pragma once

namespace saga::physics {

  /*!\brief  Base class to define an interaction
   */
  template <class TypeDescriptor, class Output,
            template <class> class... Property>
  struct interaction {

    /// Evaluate the force for two objects
    template <class Proxy>
    Output operator()(Proxy const &src, Proxy const &tgt,
                      typename TypeDescriptor::float_type delta_t) const {
      return force(delta_t,
                   Property<TypeDescriptor>::template proxy_type<
                       typename Proxy::container_type>::get(src)...,
                   Property<TypeDescriptor>::template proxy_type<
                       typename Proxy::container_type>::get(tgt)...);
    }

    /// Evaluate the force given the two sets of properties
    virtual Output force(
        typename TypeDescriptor::float_type,
        typename Property<TypeDescriptor>::underlying_value_type...,
        typename Property<TypeDescriptor>::underlying_value_type...) const = 0;
  };
} // namespace saga::physics
