#pragma once
#include "saga/particle.hpp"
#include "saga/physics/core.hpp"
#include "saga/physics/force.hpp"
#include <functional>
#include <variant>
#include <vector>

namespace saga {

  /*!\brief Class representing a collection of particles and the interactions
    among them

    This class allows to make a system of particles evolve with time after
    defining the initial locations, their momenta and the interactions among
    them.
   */
  template <class TypeDescriptor, class Properties = saga::properties<>>
  class world {

  public:
    /// Floating-point type used
    using float_type = typename TypeDescriptor::float_type;
    /// Type of the collection of functions which act as proxies for
    /// interactions
    using interactions_type =
        saga::physics::interactions_variant<TypeDescriptor>;
    /// Type of the functions which act as proxies for interactions
    using interaction_type = typename interactions_type::value_type;
    /// Type of the collection of particles
    using particles_type = saga::particles<TypeDescriptor, Properties>;
    /// Type of the collection of functions called at the end of a step
    using call_back_vector_type =
        std::vector<std::function<void(particles_type const &)>>;
    /// Type of a function called at the end of a step
    using call_back_type = typename call_back_vector_type::value_type;
    /// Type of the configuration functions
    using configuration_function_type = std::function<void(particles_type &)>;

    /// The world is constructed without arguments
    world() = default;

    /// Add a new function that will be called at the end of each step
    void add_call_back_function(call_back_type const &f) {
      m_call_back_functions.push_back(f);
    }

    /// Add a new function that will be called at the end of each step
    void add_call_back_function(call_back_type &&f) {
      m_call_back_functions.push_back(std::forward<call_back_type>(f));
    }

    /// Add a new configuration function to the world
    void configure(configuration_function_type const &f) { f(m_particles); }

    /// Add a new configuration function to the world
    void configure(configuration_function_type &&f) { f(m_particles); }

    /// Add a new interaction to the world
    template <class Interaction>
    void add_interaction(Interaction &&interaction) {
      m_interactions.emplace_back(std::forward<Interaction>(interaction));
    }

    /// Add a new interaction to the world
    template <class Interaction>
    void add_interaction(const Interaction &interaction) {
      m_interactions.emplace_back(interaction);
    }

    /// Add a new interaction to the world
    template <template <class> class Interaction, class... Args>
    void add_interaction(Args &&... args) {
      m_interactions.emplace_back(Interaction<TypeDescriptor>{args...});
    }

    /*!\brief Retrieve the set of particles

     Note that if the \ref run function has been called, the particles
     would have changed.
     */
    particles_type const &particles() const { return m_particles; }

    /// Run a series of steps using the given interval of time
    void run(std::size_t steps, float_type delta_t = 0.001) const {

      saga::physics::forces<TypeDescriptor> forces(m_particles.size());

      // execute the call-back functions at the begining of the execution
      for (auto f : m_call_back_functions)
        f(m_particles);

      for (auto s = 0u; s < steps; ++s) {

        // set all forces to zero
        for (auto f : forces)
          f.set(0.f, 0.f, 0.f, 0.f);

        for (auto i = 0u; i < m_particles.size(); ++i) {

          auto particle = m_particles[i];

          // first estimation of the positions
          particle.set_x(particle.get_x() + particle.get_px() /
                                                particle.get_mass() * 0.5 *
                                                delta_t);
          particle.set_y(particle.get_y() + particle.get_py() /
                                                particle.get_mass() * 0.5 *
                                                delta_t);
          particle.set_z(particle.get_z() + particle.get_pz() /
                                                particle.get_mass() * 0.5 *
                                                delta_t);
          particle.set_t(particle.get_t() + particle.get_e() /
                                                particle.get_mass() * 0.5 *
                                                delta_t);
        }

        // place where the point-to-point interactions are evaluated
        for (auto inter : m_interactions) {

          std::visit(
              [&](auto &&arg) -> void {
                for (auto i = 0u; i < m_particles.size(); ++i) {

                  auto pi = m_particles[i];
                  auto force_i = forces[i];

                  for (auto j = 0u; j < m_particles.size(); ++j) {

                    if (i == j)
                      continue;

                    auto pj = m_particles[j];

                    auto res = arg(pi, pj, delta_t);

                    force_i.set_x(force_i.get_x() + res.get_x());
                    force_i.set_y(force_i.get_y() + res.get_y());
                    force_i.set_z(force_i.get_z() + res.get_z());
                    force_i.set_t(force_i.get_t() + res.get_t());
                  }
                }
              },
              inter);
        }

        // integration
        for (auto i = 0u; i < m_particles.size(); ++i) {

          auto particle = m_particles[i];
          auto force = forces[i];

          auto mass =
              particle.get_mass(); // whatever we do, we must preserve the mass

          // momenta
          particle.set_px(particle.get_px() + force.get_x() * delta_t);
          particle.set_py(particle.get_py() + force.get_y() * delta_t);
          particle.set_pz(particle.get_pz() + force.get_z() * delta_t);
          particle.set_e(particle.get_e() + force.get_t() * delta_t);
          // positions
          particle.set_x(particle.get_x() + particle.get_px() /
                                                particle.get_mass() * 0.5 *
                                                delta_t);
          particle.set_y(particle.get_y() + particle.get_py() /
                                                particle.get_mass() * 0.5 *
                                                delta_t);
          particle.set_z(particle.get_z() + particle.get_pz() /
                                                particle.get_mass() * 0.5 *
                                                delta_t);
          particle.set_t(particle.get_t() + particle.get_e() /
                                                particle.get_mass() * 0.5 *
                                                delta_t);

          particle.set_mass(mass);
        }

        // call-back functions
        for (auto f : m_call_back_functions)
          f(m_particles);
      }
    }

  protected:
    /// Collection of particles in the world
    mutable particles_type m_particles;
    /// Collection of interactions
    interactions_type m_interactions;
    /// Collection of functions to be called at the end of each step
    call_back_vector_type m_call_back_functions;
  };
} // namespace saga
