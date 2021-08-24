#include "saga/physics/quantities.hpp"
#include "saga/physics/shape.hpp"

#define SAGA_DUMP_TO_FILE(file, counter)                                       \
  [&file, &counter](auto const &container) -> void {                           \
    auto n = container.size();                                                 \
                                                                               \
    for (auto i = 0u; i < n; ++i) {                                            \
                                                                               \
      auto p = container[i];                                                   \
                                                                               \
      if constexpr (p.template has<saga::physics::radius>()) {                 \
        file << counter << ' ' << (p.template get<saga::property::x>()) << ' ' \
             << (p.template get<saga::property::y>()) << ' '                   \
             << (p.template get<saga::property::z>()) << ' '                   \
             << (p.template get<saga::property::px>()) << ' '                  \
             << (p.template get<saga::property::py>()) << ' '                  \
             << (p.template get<saga::property::pz>()) << ' '                  \
             << (p.template get<saga::physics::radius>()) << std::endl;        \
      } else {                                                                 \
        file << counter << ' ' << (p.template get<saga::property::x>()) << ' ' \
             << (p.template get<saga::property::y>()) << ' '                   \
             << (p.template get<saga::property::z>()) << ' '                   \
             << (p.template get<saga::property::px>()) << ' '                  \
             << (p.template get<saga::property::py>()) << ' '                  \
             << (p.template get<saga::property::pz>()) << ' ' << 1.f           \
             << std::endl;                                                     \
      }                                                                        \
    }                                                                          \
                                                                               \
    ++counter;                                                                 \
  }
