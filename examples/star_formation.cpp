#include "dump.hpp"
#include "saga/all.hpp"
#include <fstream>
#include <iostream>
#include <random>

using sou = saga::solar_system<saga::types::cpu::single_float_precision>;

int main() {

  saga::world<saga::types::cpu::single_float_precision, saga::physics::sphere>
      world;

  world.set_collision_handler<saga::physics::collision::simple_merge>();

  auto delta_t = sou::time_from_si(30.f * 24.f * 3600.f);

  world.add_interaction<
      saga::physics::gravitational_non_relativistic_interaction>(
      sou::gravitational_constant);

  world.configure([&](auto &container) {
    container.resize(10);

    std::random_device
        rd; // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    std::uniform_real_distribution<typename sou::float_type> uniform(
        -200., +200.); // Gm

    for (auto p : container) {

      p.set_x(uniform(gen));
      p.set_y(uniform(gen));
      p.set_z(uniform(gen));
      p.set_t(0.f);

      // static particles
      p.set_px(0.f);
      p.set_py(0.f);
      p.set_pz(0.f);
      p.set_e(1e-1); // Mo

      p.template set<saga::physics::radius>(1.f); // Gm
    }
  });

  std::ofstream file{"star_formation.txt"};
  std::size_t counter = 0;
  world.add_call_back_function(SAGA_DUMP_TO_FILE(file, counter));

  world.run(100 * 12, delta_t);
}
