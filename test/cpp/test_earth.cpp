#include "saga/all.hpp"
#include <fstream>
#include <iostream>

int main() {

  saga::world<saga::types::cpu::single_float_precision> world;

  using sou = saga::solar_system<saga::types::cpu::single_float_precision>;

  auto delta_t = sou::time_from_si(24.f * 3600.f);

  world.add_interaction<saga::gravitational_non_relativistic_interaction>(
      sou::gravitational_constant);

  world.configure([&](auto &container) {
    container.resize(3);

    auto sun = container[0];
    sun.set_x(0);
    sun.set_y(0);
    sun.set_z(0);
    sun.set_t(0);

    sun.set_px(0);
    sun.set_py(0);
    sun.set_pz(0);
    sun.set_e(sou::sun_mass); // Mo * (Mm / s)^2

    auto earth = container[1];
    earth.set_x(sou::earth_perihelion); // Mm
    earth.set_y(0);
    earth.set_z(0);
    earth.set_t(0);

    earth.set_px(0);
    earth.set_py(sou::earth_mass * sou::earth_perihelion_velocity); // Mo * c
    earth.set_pz(0);
    earth.set_e(sou::earth_mass +
                0.5 * sou::earth_mass * sou::earth_perihelion_velocity *
                    sou::earth_perihelion_velocity); // Mo * c^2

    auto mars = container[2];
    mars.set_x(earth.get_px() + sou::mars_perihelion); // Mm
    mars.set_y(0);
    mars.set_z(0);
    mars.set_t(0);

    mars.set_px(0);
    mars.set_py(sou::mars_mass * sou::mars_perihelion_velocity); // Mo * c
    mars.set_pz(0);
    mars.set_e(sou::mars_mass + 0.5 * sou::mars_mass *
                                    sou::mars_perihelion_velocity *
                                    sou::mars_perihelion_velocity); // Mo * c^2
  });

  std::ofstream file;
  file.open("solar_system.txt");

  world.add_call_back_function([&file](auto const &container) {
    for (auto p : container) {

      file << (p.template get<saga::property::x>()) << ' '
           << (p.template get<saga::property::y>()) << ' '
           << (p.template get<saga::property::z>()) << ' ';
    }

    file << std::endl;
  });

  world.run(5 * 365, delta_t); // 3 years
}
