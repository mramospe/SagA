#include "saga/all.hpp"
#include "test_utils.hpp"

// Alias functions to test a container of forces
saga::test::errors test_forces_container() {
  return saga::test::test_container<saga::core::forces<saga::cpu::sf>>();
}

saga::test::errors test_forces_proxy() {

  using container_of_forces = saga::core::forces<saga::cpu::sf>;

  auto errors = saga::test::test_iterator<container_of_forces>();

  container_of_forces container(1);

  auto p = container[0];

  p.get<saga::property::x>();

  return errors;
}

saga::test::errors test_forces_value() {
  return saga::test::test_value<saga::core::forces<saga::cpu::sf>>();
}

// Alias functions to test a container of particles
saga::test::errors test_particles_container() {
  return saga::test::test_container<saga::particles<saga::cpu::sf>>();
}

saga::test::errors test_particles_proxy() {

  using container_of_particles = saga::particles<saga::cpu::sf>;

  auto errors = saga::test::test_iterator<container_of_particles>();

  container_of_particles container(1);

  auto p = container[0];

  p.get<saga::property::x>();

  return errors;
}

saga::test::errors test_particles_value() {
  return saga::test::test_value<saga::particles<saga::cpu::sf>>();
}

saga::test::errors test_particles_backend() {

  saga::test::errors errors;

  using cpu_particles_type = saga::particles<saga::cpu::sf>;
  using cuda_particles_type =
      typename cpu_particles_type::type_with_backend<saga::backend::CUDA>;

  if (!std::is_same_v<cuda_particles_type, saga::particles<saga::cuda::sf>>)
    errors.emplace_back(
        "Unable to switch the particle container backend between CPU and CUDA");

  return errors;
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
  SAGA_TEST_UTILS_ADD_TEST(particles_collector, test_particles_backend);

  auto forces_status = forces_collector.run();
  auto particles_status = particles_collector.run();

  return !(forces_status && particles_status);
}
