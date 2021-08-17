#pragma once
#include "saga/physics/quantities.hpp"
#include <fstream>
#include <string>

#ifndef REACTIONS_TEST_UTILS_HPP
#define REACTIONS_TEST_UTILS_HPP

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace saga::test {

  /// Container of errors
  using errors = std::vector<std::string>;

  /* \brief Object handling a collection of test functions
   *
   */
  class collector {

  public:
    /// Functions must return a list with the error messages
    using function_type = std::function<errors(void)>;

    /// Constructor from a name
    collector(std::string const &name) : m_name{name} {}
    /// Default destructor
    ~collector() = default;

    // Invalidate other constructors
    collector(collector const &) = delete;
    collector(collector &&) = delete;
    collector &operator=(collector const &) = delete;

    /// Add a new test function
    void add_test_function(std::string name, function_type const &function) {
      m_functions.push_back(std::make_pair(std::move(name), function));
    }

    /// Run the stored tests and return the status
    bool run() const {

      std::map<std::size_t, std::vector<std::string>> error_map;

      for (auto i = 0u; i < m_functions.size(); ++i) {
        auto v = m_functions[i].second();
        if (v.size() != 0)
          error_map[i] = std::move(v);
      }

      std::cout << "Results for collector \"" << m_name << '\"' << std::endl;
      for (auto i = 0u; i < m_functions.size(); ++i)
        std::cout << "- "
                  << (error_map.find(i) == error_map.cend() ? "(success) "
                                                            : "(failed) ")
                  << m_functions[i].first << std::endl;

      if (error_map.size() != 0) {
        std::cerr << "summary of errors:" << std::endl;
        for (auto const &p : error_map) {
          std::cerr << "* " << m_functions[p.first].first << ':' << std::endl;
          for (auto const &what : p.second)
            std::cerr << " - " << what << std::endl;
        }
        return false;
      }

      return true;
    }

  protected:
    /// Names and functions
    std::vector<std::pair<std::string, function_type>> m_functions;
    /// Name of the collector
    std::string m_name;
  };

  // Template function to test a container
  template <class Container> saga::test::errors test_container() {

    saga::test::errors errors;

    Container container;
    if (container.size() != 0)
      errors.emplace_back("Container constructor does not default to empty");

    container.resize(10);
    if (container.size() != 10)
      errors.emplace_back("Failed to resize container");

    Container container_from_size(10);
    if (container_from_size.size() != 10)
      errors.emplace_back("Failed to build container for a given size");

    return errors;
  }

  // Template function to test proxies
  template <class Container> saga::test::errors test_proxy() {

    saga::test::errors errors;

    Container container(10);

    auto proxy_0 = container[0];
    auto proxy_1 = container[1];

    proxy_0 = proxy_1;
    if (proxy_0.index() == proxy_1.index())
      errors.emplace_back(
          "Assigning a proxy to another is setting the indices");

    auto cproxy_1 = container.cbegin();
    auto cproxy_0 = cproxy_1++;
    if (cproxy_0.index() == cproxy_1.index())
      errors.emplace_back(
          "Assigning a constant proxy to another is setting the indices");

    if (container.begin() == container.end())
      errors.emplace_back(
          "Beginning and end of container evaluate to the same proxy");

    if (container.cbegin() == container.cend())
      errors.emplace_back(
          "Beginning and end of constant container evaluate to the same proxy");

    return errors;
  }

  // Template function to test values
  template <class Container> saga::test::errors test_value() {

    saga::test::errors errors;

    Container container(10);

    typename Container::value_type value;

    container[0] = value;

    auto proxy = container[1];

    proxy = value;
    value = container[1];
    value = container.cbegin();

    return errors;
  }
} // namespace saga::test

#define SAGA_TEST_UTILS_ADD_TEST(collector, function)                          \
  collector.add_test_function(#function, function);

#define SAGA_TEST_UTILS_CATCH_EXCEPTIONS(errors)                               \
  catch (...) {                                                                \
    errors.push_back("Unknown error detected");                                \
  }

#endif // SAGA_TEST_UTILS_HPP
