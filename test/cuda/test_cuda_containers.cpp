#include "saga/all.hpp"
#include "test_utils.hpp"

#define SAGA_TEST_DEFAULT_FLOAT_VALUE 1.f

struct set_particle {
  template <class Proxy> __device__ void operator()(Proxy proxy) const {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    proxy.set_momenta_and_mass(gtid, gtid, gtid, SAGA_TEST_DEFAULT_FLOAT_VALUE);
  }
};

struct set_value {
  template <class View> __device__ void operator()(View &view) const {
    view = SAGA_TEST_DEFAULT_FLOAT_VALUE;
  }
};

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
    p.set_momenta_and_mass(0.f, 0.f, 0.f, SAGA_TEST_DEFAULT_FLOAT_VALUE);

  for (auto p : particles) {
    if (!saga::test::is_close(p.get_mass(), SAGA_TEST_DEFAULT_FLOAT_VALUE)) {
      errors.push_back("Problems assigning values with proxies in the host");
      return errors;
    }
  }

  // check that the values are the same after passing the data to the device and
  // back to the host
  auto cuda_particles = saga::to_backend<saga::backend::CUDA>(particles);

  particles = saga::to_backend<saga::backend::CPU>(cuda_particles);

  for (auto p : particles) {
    if (!saga::test::is_close(p.get_mass(), SAGA_TEST_DEFAULT_FLOAT_VALUE)) {
      errors.push_back("Problems assigning values with proxies in the host");
      return errors;
    }
  }

  return errors;
}

saga::test::errors test_vector_iteration() {

  saga::test::errors errors;

  using vector_type = saga::core::vector<float, saga::backend::CUDA>;

  vector_type vector(10);

  try {

    auto vector_view = saga::core::make_vector_view(vector);

    saga::core::apply_functor(vector_view, set_value{});

  } catch (std::runtime_error const &e) {
    errors.push_back(e.what());
    return errors;
  }

  try {
    auto cpu_vector = saga::core::to_host(vector);

    for (auto i = 0u; i < cpu_vector.size(); ++i) {
      if (!saga::test::is_close(cpu_vector[i], SAGA_TEST_DEFAULT_FLOAT_VALUE)) {
        errors.push_back("Problems setting the values of a vector using CUDA");
        break;
      }
    }
  } catch (std::runtime_error const &e) {
    errors.push_back(e.what());
    return errors;
  }

  return errors;
}

// Test that we can iterate the container
saga::test::errors test_container_iteration() {

  saga::test::errors errors;

  using particles_type = saga::particles<saga::cuda::sf>;

  particles_type particles(10);

  try {

    auto particles_view = saga::core::make_container_view(particles);

    saga::core::apply_functor(particles_view, set_particle{});

  } catch (std::runtime_error const &e) {
    errors.push_back(e.what());
    return errors;
  }

  try {
    auto cpu_particles = saga::to_backend<saga::backend::CPU>(particles);

    for (auto i = 0u; i < cpu_particles.size(); ++i) {

      auto const proxy = cpu_particles[i];

      if (!saga::test::is_close(proxy.get_px(), i) ||
          !saga::test::is_close(proxy.get_py(), i) ||
          !saga::test::is_close(proxy.get_pz(), i) ||
          !saga::test::is_close(proxy.get_mass(),
                                SAGA_TEST_DEFAULT_FLOAT_VALUE)) {
        std::cout << proxy.get_px() << ' ' << proxy.get_py() << ' '
                  << proxy.get_pz() << ' ' << proxy.get_mass() << std::endl;
        errors.push_back("Error assigning values to the particles");
        break;
      }
    }
  } catch (std::runtime_error const &e) {
    errors.push_back(e.what());
    return errors;
  }

  return errors;
}

int main() {

  saga::test::collector collector("container");
  SAGA_TEST_UTILS_ADD_TEST(collector, test_container);
  SAGA_TEST_UTILS_ADD_TEST(collector, test_backend);
  SAGA_TEST_UTILS_ADD_TEST(collector, test_vector_iteration);
  SAGA_TEST_UTILS_ADD_TEST(collector, test_container_iteration);

  return !collector.run();
}
