#pragma once
#include "saga/physics/quantities.hpp"
#include <fstream>
#include <string>

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
  template <class Container> saga::test::errors test_iterator() {

    saga::test::errors errors;

    Container container(10);

    auto it_0 = container.begin();
    auto it_1 = container.begin() + 1;

    it_0 = it_1;
    if (it_0 != it_1)
      errors.emplace_back(
          "Assigning an iterator to another is not done correctly");

    auto cit_0 = container.cbegin();
    auto cit_1 = ++cit_0;

    if (cit_0 != cit_1)
      errors.emplace_back(
          "Assigning a constant iterator to another is not done correctly");

    if (container.begin() == container.end())
      errors.emplace_back(
          "Beginning and end of container evaluate to the same iterator");

    if (container.cbegin() == container.cend())
      errors.emplace_back("Beginning and end of constant container evaluate to "
                          "the same iterator");

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
    value = *(container.cbegin());

    return errors;
  }
} // namespace saga::test

#define SAGA_TEST_UTILS_ADD_TEST(collector, function)                          \
  collector.add_test_function(#function, function);

#define SAGA_TEST_UTILS_CATCH_EXCEPTIONS(errors)                               \
  catch (...) {                                                                \
    errors.push_back("Unknown error detected");                                \
  }
