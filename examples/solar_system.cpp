#include "dump.hpp"
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

  planet.template set<saga::physics::radius>(Planet::radius);
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

  sun.template set<saga::physics::radius>(sou::sun::radius);

  prepare_for_planets_impl<Planet...>(
      container, std::make_index_sequence<sizeof...(Planet)>());
}

int main() {

  saga::world<saga::types::cpu::single_float_precision, saga::physics::sphere>
      world;

  auto delta_t = sou::time_from_si(24.f * 3600.f);

  world.add_interaction<
      saga::physics::gravitational_non_relativistic_interaction>(
      saga::physics::field_constant<saga::types::cpu::single_float_precision>{
          sou::gravitational_constant});

  world.configure([&](auto &container) {
    prepare_for_planets<sou::mercury, sou::venus, sou::earth, sou::mars,
                        sou::jupiter, sou::saturn, sou::uranus, sou::neptune>(
        container);
  });

  std::ofstream file{"solar_system.txt"};

  std::size_t counter = 0;
  world.add_call_back_function(SAGA_DUMP_TO_FILE(file, counter));

  world.run(5 * 365, delta_t); // 5 years
}
