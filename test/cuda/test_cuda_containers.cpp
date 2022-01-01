#include "saga/all.hpp"
#include "test_utils.hpp"

template <class Proxy> __device__ void set_particle(Proxy proxy) {

  auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

  proxy.set_momenta_and_mass(gtid, gtid, gtid, 1.);
}

// Test the creation of a container
saga::test::errors test_container() {
  return saga::test::test_container<saga::particles<saga::cuda::sf>>();
}

// Test that switching backends maintains the values
saga::test::errors test_backend() {

  saga::test::errors errors;

  // check that the proxy is working on the host
  saga::particles<saga::cpu::sf> particles(10);

  for (auto p : particles)
    p.set_momenta_and_mass(0.f, 0.f, 0.f, 1.f);

  for (auto p : particles) {
    if (!saga::test::is_close(p.get_mass(), 1.f)) {
      errors.push_back("Problems assigning values with proxies in the host");
      break;
    }
  }

  // check that the values are the same after passing the data to the device and
  // back to the host
  auto cuda_particles = saga::to_backend<saga::backend::CUDA>(particles);

  particles = saga::to_backend<saga::backend::CPU>(cuda_particles);

  for (auto p : particles) {
    if (!saga::test::is_close(p.get_mass(), 1.f)) {
      errors.push_back("Problems assigning values with proxies in the host");
      break;
    }
  }

  return errors;
}

// Test that we can iterate the container
saga::test::errors test_iteration() {

  saga::test::errors errors;

  using particles_type = saga::particles<saga::cuda::sf>;

  particles_type particles(10);

  saga::core::cuda::apply_simple_function_inplace(
      particles, &set_particle<typename particles_type::proxy_type>);

  auto cpu_particles = saga::to_backend<saga::backend::CPU>(particles);

  for (auto i = 0u; i < cpu_particles.size(); ++i) {

    auto const proxy = cpu_particles[i];

    if (!saga::test::is_close(proxy.get_px(), i) ||
        !saga::test::is_close(proxy.get_py(), i) ||
        !saga::test::is_close(proxy.get_pz(), i) ||
        !saga::test::is_close(proxy.get_mass(), 1.f)) {
      std::cout << proxy.get_px() << ' ' << proxy.get_py() << ' '
                << proxy.get_pz() << ' ' << proxy.get_mass() << std::endl;
      errors.push_back("Error assigning values to the particles");
      break;
    }
  }

  return errors;
}

int main() {

  saga::test::collector collector("container");
  SAGA_TEST_UTILS_ADD_TEST(collector, test_container);
  SAGA_TEST_UTILS_ADD_TEST(collector, test_backend);
  SAGA_TEST_UTILS_ADD_TEST(collector, test_iteration);

  return !collector.run();
}
