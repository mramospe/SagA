#include "saga/all.hpp"
#include <fstream>

using sou = saga::earth_system<saga::types::cpu::single_float_precision>;

int main() {

  saga::world<saga::types::cpu::single_float_precision> world;

  auto delta_t = sou::time_from_si(3600.);

  world.add_interaction<
      saga::physics::gravitational_non_relativistic_interaction>(
      sou::gravitational_constant);

  world.configure([&](auto &container) {
    container.resize(2);

    auto earth = container[0];
    earth.set_x(0);
    earth.set_y(0);
    earth.set_z(0);
    earth.set_t(0);

    earth.set_px(0);
    earth.set_py(0);
    earth.set_pz(0);
    earth.set_e(sou::earth::mass);

    auto moon = container[1];
    moon.set_x(sou::moon::perigee);
    moon.set_y(0);
    moon.set_z(0);
    moon.set_t(0);

    moon.set_px(0);
    moon.set_py(sou::moon::mass * sou::moon::perigee_velocity);
    moon.set_pz(0);
    moon.set_e(sou::moon::mass + 0.5 * sou::moon::mass *
                                     sou::moon::perigee_velocity *
                                     sou::moon::perigee_velocity);
  });

  std::ofstream file;
  file.open("earth_and_moon.txt");

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
