#include "saga/all.hpp"
#include "test_utils.hpp"

// Alias functions to test a container of forces
saga::test::errors test_forces_container() {
  return saga::test::test_container<saga::physics::forces<saga::cpu::sf>>();
}

saga::test::errors test_forces_proxy() {

  using container_of_forces = saga::physics::forces<saga::cpu::sf>;

  auto errors = saga::test::test_proxy<container_of_forces>();

  container_of_forces container(1);

  auto p = container[0];

  p.get<saga::property::x>();

  return errors;
}

saga::test::errors test_forces_value() {
  return saga::test::test_value<saga::physics::forces<saga::cpu::sf>>();
}

// Alias functions to test a container of particles
saga::test::errors test_particles_container() {
  return saga::test::test_container<saga::particles<saga::cpu::sf>>();
}

saga::test::errors test_particles_proxy() {

  using container_of_particles = saga::particles<saga::cpu::sf>;

  auto errors = saga::test::test_proxy<container_of_particles>();

  container_of_particles container(1);

  auto p = container[0];

  p.get<saga::property::x>();

  return errors;
}

saga::test::errors test_particles_value() {
  return saga::test::test_value<saga::particles<saga::cpu::sf>>();
}

int main() {

  saga::test::collector forces_collector("forces");
  SAGA_TEST_UTILS_ADD_TEST(forces_collector, test_forces_container);
  SAGA_TEST_UTILS_ADD_TEST(forces_collector, test_forces_proxy);
  SAGA_TEST_UTILS_ADD_TEST(forces_collector, test_forces_value);

  saga::test::collector particles_collector("particles");
  SAGA_TEST_UTILS_ADD_TEST(particles_collector, test_particles_container);
  SAGA_TEST_UTILS_ADD_TEST(particles_collector, test_particles_proxy);
  SAGA_TEST_UTILS_ADD_TEST(particles_collector, test_particles_value);

  auto forces_status = forces_collector.run();
  auto particles_status = particles_collector.run();

  return !(forces_status && particles_status);
}
