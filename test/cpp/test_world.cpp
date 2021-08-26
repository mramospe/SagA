#include "saga/all.hpp"
#include "test_utils.hpp"

// Test particles subject only to the gravitational interaction
saga::test::errors test_gravitational() {

  saga::test::errors errors;

  saga::world<saga::types::cpu::single_float_precision> world;

  std::size_t particles_per_coordinate = 2;
  std::size_t number_of_particles = particles_per_coordinate *
                                    particles_per_coordinate *
                                    particles_per_coordinate;

  world.add_interaction<
      saga::physics::gravitational_non_relativistic_interaction>(
      saga::physics::field_constant<saga::types::cpu::single_float_precision>{
          1.f});

  world.configure([&](auto &container) {
    container.resize(number_of_particles);

    for (auto i = 0u; i < container.size(); ++i) {

      auto p = container[i];

      p.set_x(-0.5 + i / (particles_per_coordinate * particles_per_coordinate));
      p.set_y(-0.5 + (i / particles_per_coordinate) % particles_per_coordinate);
      p.set_z(-0.5 + i % particles_per_coordinate);
      p.set_t(0);
      p.set_px(0);
      p.set_py(0);
      p.set_pz(0);
      p.set_e(1000);
    }
  });

  if (world.particles().size() != number_of_particles)
    errors.emplace_back(
        "World size is not correctly set by the configuration function");

  world.run(1000, 0.01);

  return errors;
}

// Test elastic collisions of particles subject to gravity
saga::test::errors test_collisions_elastic() {

  saga::test::errors errors;

  saga::world<saga::types::cpu::single_float_precision, saga::physics::sphere>
      world;

  world.set_collision_handler<saga::physics::collision::elastic>();

  std::size_t particles_per_coordinate = 2;
  std::size_t number_of_particles = particles_per_coordinate *
                                    particles_per_coordinate *
                                    particles_per_coordinate;

  world.add_interaction<
      saga::physics::gravitational_non_relativistic_interaction>(
      saga::physics::field_constant<saga::types::cpu::single_float_precision>{
          1.f});

  world.configure([&](auto &container) {
    container.resize(number_of_particles);

    for (auto i = 0u; i < container.size(); ++i) {

      auto p = container[i];

      p.set_x(-0.5 + i / (particles_per_coordinate * particles_per_coordinate));
      p.set_y(-0.5 + (i / particles_per_coordinate) % particles_per_coordinate);
      p.set_z(-0.5 + i % particles_per_coordinate);
      p.set_t(0);
      p.set_px(0);
      p.set_py(0);
      p.set_pz(0);
      p.set_e(1000);
      p.template set<saga::physics::radius>(0.1f);
    }
  });

  world.run(1000, 0.01);

  return errors;
}

// Test collisions of particles subject to gravity with simple merges
saga::test::errors test_collisions_simple_merge() {

  saga::test::errors errors;

  saga::world<saga::types::cpu::single_float_precision, saga::physics::sphere>
      world;

  world.set_collision_handler<saga::physics::collision::simple_merge>();

  std::size_t particles_per_coordinate = 2;
  std::size_t number_of_particles = particles_per_coordinate *
                                    particles_per_coordinate *
                                    particles_per_coordinate;

  world.add_interaction<
      saga::physics::gravitational_non_relativistic_interaction>(
      saga::physics::field_constant<saga::types::cpu::single_float_precision>{
          1.f});

  world.configure([&](auto &container) {
    container.resize(number_of_particles);

    for (auto i = 0u; i < container.size(); ++i) {

      auto p = container[i];

      p.set_x(-0.5 + i / (particles_per_coordinate * particles_per_coordinate));
      p.set_y(-0.5 + (i / particles_per_coordinate) % particles_per_coordinate);
      p.set_z(-0.5 + i % particles_per_coordinate);
      p.set_t(0);
      p.set_px(0);
      p.set_py(0);
      p.set_pz(0);
      p.set_e(1000);
      p.template set<saga::physics::radius>(0.1f);
    }
  });

  world.run(1000, 0.01);

  return errors;
}

int main() {

  saga::test::collector world_collector("world");
  SAGA_TEST_UTILS_ADD_TEST(world_collector, test_gravitational);

  saga::test::collector collisions_collector("world");
  SAGA_TEST_UTILS_ADD_TEST(collisions_collector, test_collisions_elastic);
  SAGA_TEST_UTILS_ADD_TEST(collisions_collector, test_collisions_simple_merge);

  auto world_status = world_collector.run();
  auto collisions_status = collisions_collector.run();

  return !world_status && !collisions_status;
}
