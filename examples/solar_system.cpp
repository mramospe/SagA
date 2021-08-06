#include "saga/all.hpp"
#include <fstream>

using sou = saga::solar_system<saga::types::cpu::single_float_precision>;

template <class Planet, std::size_t I, class Particles>
void prepare_planet(Particles &container) {

  auto planet = container[I];

  planet.set_x(Planet::perihelion);
  planet.set_y(0);
  planet.set_z(0);
  planet.set_t(0);

  planet.set_px(0);
  planet.set_py(Planet::mass * Planet::perihelion_velocity);
  planet.set_pz(0);
  planet.set_e(Planet::mass + 0.5 * Planet::mass * Planet::perihelion_velocity *
                                  Planet::perihelion_velocity);
}

template <class... Planet, class Particles, std::size_t... I>
void prepare_for_planets_impl(Particles &container, std::index_sequence<I...>) {

  (prepare_planet<Planet, I + 1>(container), ...);
}

template <class... Planet, class Particles>
void prepare_for_planets(Particles &container) {

  container.resize(1 + sizeof...(Planet));

  auto sun = container[0];
  sun.set_x(0);
  sun.set_y(0);
  sun.set_z(0);
  sun.set_t(0);

  sun.set_px(0);
  sun.set_py(0);
  sun.set_pz(0);
  sun.set_e(sou::sun::mass);

  prepare_for_planets_impl<Planet...>(
      container, std::make_index_sequence<sizeof...(Planet)>());
}

int main() {

  saga::world<saga::types::cpu::single_float_precision> world;

  auto delta_t = sou::time_from_si(24.f * 3600.f);

  world.add_interaction<
      saga::physics::gravitational_non_relativistic_interaction>(
      sou::gravitational_constant);

  world.configure([&](auto &container) {
    prepare_for_planets<sou::mercury, sou::venus, sou::earth, sou::mars,
                        sou::jupiter, sou::saturn, sou::uranus, sou::neptune>(
        container);
  });

  std::ofstream file;
  file.open("solar_system.txt");

  world.add_call_back_function([&file](auto const &container) {
    auto n = container.size();

    for (auto i = 0u; i < n; ++i) {

      auto p = container[i];

      if (i > 0)
        file << ' ';

      file << (p.template get<saga::property::x>()) << ' '
           << (p.template get<saga::property::y>()) << ' '
           << (p.template get<saga::property::z>());
    }

    file << std::endl;
  });

  world.run(5 * 365, delta_t); // 3 years
}
