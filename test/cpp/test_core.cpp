#include "saga/all.hpp"
#include "test_utils.hpp"

using simple_container =
    saga::core::container_with_fields<saga::types::cpu::single_float_precision,
                                      saga::property::x, saga::property::y>;

saga::test::errors test_container() {

  auto errors = saga::test::test_container<simple_container>();

  simple_container container(10);

  if (container.size() != container.get<saga::property::x>().size())
    errors.emplace_back(
        "Size of the top-level container and internal container differ");

  if (container.get<saga::property::x>().size() !=
      container.get<saga::property::y>().size())
    errors.emplace_back("Sizes of internal containers differ");

  container.resize(2);

  if (container.size() != container.get<saga::property::x>().size())
    errors.emplace_back(
        "Size of the top-level container and internal container differ");

  if (container.get<saga::property::x>().size() !=
      container.get<saga::property::y>().size())
    errors.emplace_back("Sizes of internal containers differ");

  return errors;
}

saga::test::errors test_proxy() {

  auto errors = saga::test::test_proxy<simple_container>();

  simple_container container(10);

  auto p1 = container[0];
  p1.get<saga::property::x>();

  auto p2 = p1;
  if (p1.index() != p2.index())
    errors.emplace_back("Unable to construct proxy from assignment properly");

  ++p1;
  p2 = p1;
  if (p1.index() == p2.index())
    errors.emplace_back(
        "Assignment without construction must leave the index untouched");

  auto idx = p1.index();
  if ((++p1).index() == idx)
    errors.emplace_back("Increment operator does not modify the index");

  if ((++container[0]).index() != container[1].index())
    errors.emplace_back("Error in proxy increment operator");

  if ((container[0]++).index() != container[0].index())
    errors.emplace_back("Error in proxy increment operator (copy)");

  if (container.begin() == container.end())
    errors.emplace_back("Issues with the indices of proxies");

  if (container.cbegin() == container.cend())
    errors.emplace_back("Issues with the indices of constant proxies");

  if (container.begin() != container.cbegin())
    errors.emplace_back(
        "Issues with the indices of proxies and constant proxies");

  return errors;
}

saga::test::errors test_value() {

  auto errors = saga::test::test_value<simple_container>();

  simple_container container(10);

  typename simple_container::value_type v = container[0];
  v.get<saga::property::x>();

  return errors;
}

int main() {

  saga::test::collector container_collector("container");
  SAGA_TEST_UTILS_ADD_TEST(container_collector, test_container);
  SAGA_TEST_UTILS_ADD_TEST(container_collector, test_proxy);
  SAGA_TEST_UTILS_ADD_TEST(container_collector, test_value);

  return !container_collector.run();
}
