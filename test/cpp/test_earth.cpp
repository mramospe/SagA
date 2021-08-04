#include "saga/world.hpp"
#include <fstream>
#include <iostream>

int main() {

  saga::world<saga::types::cpu::single_float_precision> world;

  using float_type = saga::types::cpu::single_float_precision::float_type;

  float_type speed_of_light = 300.f;               // Mm * s^-1
  float_type sun_mass = 1.;                        // Mo
  float_type gravitational_constant = 0.00147462;  // Mo^-1 * Mm * c^2
  float_type earth_mass = 3.003e-6;                // Mo
  float_type delta_t_in_days = 1.;                 // days
  float_type distance_from_earth_to_sun = 1.496e5; // Mm
  float_type earth_velocity = 0.030f;              // Mm * s^-1

  float_type delta_t = speed_of_light * delta_t_in_days * 24.f * 3600.f;

  world.add_interaction<saga::gravitational_non_relativistic_interaction>(
      gravitational_constant);

  world.configure([&](auto &container) {
    container.resize(2);

    auto sun = container[0];
    sun.set_x(0);
    sun.set_y(0);
    sun.set_z(0);
    sun.set_t(0);

    sun.set_px(0);
    sun.set_py(0);
    sun.set_pz(0);
    sun.set_e(sun_mass); // Mo * (Mm / s)^2

    auto earth = container[1];
    earth.set_x(distance_from_earth_to_sun); // Mm
    earth.set_y(0);
    earth.set_z(0);
    earth.set_t(0);

    earth.set_px(0);
    earth.set_py(earth_mass * earth_velocity / speed_of_light); // Mo * c
    earth.set_pz(0);
    earth.set_e(earth_mass + 0.5 * earth_mass * earth_velocity *
                                 earth_velocity /
                                 (speed_of_light * speed_of_light)); // Mo * c^2
  });

  std::ofstream file;
  file.open("data_earth.txt");

  world.add_call_back_function([&file](auto const &container) {
    for (auto p : container) {

      file << (p.template get<saga::property::x>()) << ' '
           << (p.template get<saga::property::y>()) << ' '
           << (p.template get<saga::property::z>()) << ' ';
    }

    file << std::endl;
  });

  world.run(3 * 365, delta_t); // 3 years
}
