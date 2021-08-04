#include "saga/all.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

template <class Container> void function(Container const &c) { c[0]; }

int main() {

  saga::particles<saga::types::cpu::single_float_precision> particles;
  particles.reserve(10);
  std::cout << particles.size() << std::endl;
  saga::particle<saga::types::cpu::single_float_precision> particle;
  particles.push_back(particle);
  particles.push_back(std::move(particle));
  std::cout << particles.size() << std::endl;
  auto p0 = particles[0];
  std::cout << p0.get<saga::property::x>() << std::endl;
  function(particles);
  particles.push_back(particles[1]);

  saga::world<saga::types::cpu::single_float_precision> world;

  world.configure([](auto &container) -> void {
    auto n = 2u;

    container.resize(n * n * n);

    for (auto i = 0u; i < container.size(); ++i) {

      auto p = container[i];

      p.set_x(-0.5 + i / (n * n));
      p.set_y(-0.5 + (i / n) % n);
      p.set_z(-0.5 + i % n);
      p.set_t(0);
      p.set_px(0);
      p.set_py(0);
      p.set_pz(0);
      p.set_e(1000);
    }
  });

  std::cout << "particles: " << world.particles().size() << std::endl;

  std::cout << std::boolalpha << (particles[0] == particles[1]) << ' '
            << (particles[0] == particles[0]) << std::endl;

  std::cout << "-- Initial --" << std::endl;
  for (auto i = 0u;
       i < (world.particles().size() < 25u ? world.particles().size() : 25u);
       ++i)
    std::cout << (world.particles()[i].get<saga::property::x>()) << ' '
              << (world.particles()[i].get<saga::property::y>()) << ' '
              << (world.particles()[i].get<saga::property::z>()) << ' '
              << (world.particles()[i].get<saga::property::px>()) << ' '
              << (world.particles()[i].get<saga::property::py>()) << ' '
              << (world.particles()[i].get<saga::property::pz>()) << ' '
              << (world.particles()[i].get<saga::property::e>()) << ' '
              << (world.particles()[i].get_mass()) << std::endl;

  world.add_interaction<saga::gravitational_non_relativistic_interaction>();

  std::ofstream file;
  file.open("data.txt");

  world.add_call_back_function([&file](auto const &container) -> void {
    for (auto p : container) {

      file << (p.template get<saga::property::x>()) << ' '
           << (p.template get<saga::property::y>()) << ' '
           << (p.template get<saga::property::z>()) << ' ';
    }

    file << std::endl;
  });

  world.run(500000, 0.01);

  std::cout << "-- Final --" << std::endl;
  for (auto i = 0u;
       i < (world.particles().size() < 25u ? world.particles().size() : 25u);
       ++i)
    std::cout << std::setprecision(3) << std::scientific
              << world.particles().get<saga::property::x>(i) << ' '
              << world.particles().get<saga::property::y>(i) << ' '
              << world.particles().get<saga::property::z>(i) << ' '
              << world.particles().get<saga::property::px>(i) << ' '
              << world.particles().get<saga::property::py>(i) << ' '
              << world.particles().get<saga::property::pz>(i) << ' '
              << std::endl;

  file.close();

  return 0;
}
