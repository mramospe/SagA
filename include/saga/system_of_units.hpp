#pragma once

namespace saga {

  /*!\brief System of units for to use in calculations at the solar system scale

    Units are provided in the following way:

    * Masses referenced to the solar mass
    * The speed of light is set to unity
    * Distances are measured in thousands of kilometers

    Since the definition of distance, time and speed of light are linked
    together, and the latter is set to one, maintaining the definition of the
    distance constant implies we have to change the definition of seconds, which
    must be provided in units of seconds / speed of light (thus multiplying the
    amount of seconds by the speed of light).
   */
  template <class TypeDescriptor> struct solar_system {

    using float_type = typename TypeDescriptor::float_type;

    // Constant
    static constexpr float_type gravitational_constant =
        0.00147462; // Mo^-1 * Mm * c^2
    // Masses
    static constexpr float_type sun_mass = 1.;                     // Mo
    static constexpr float_type earth_mass = 3.003e-6;             // Mo
    static constexpr float_type mars_mass = 0.107 * earth_mass;    // Mo
    static constexpr float_type moon_mass = 0.012300 * earth_mass; // Mo
    // Distances
    static constexpr float_type earth_perihelion = 147.10e3;           // Mm
    static constexpr float_type mars_perihelion = 206.7e3;             // Mm
    static constexpr float_type distance_from_moon_to_earth = 384.400; // Mm
    // Velocities
    static constexpr float_type speed_of_light = 299.792458; // Mm * s^-1
    static constexpr float_type earth_perihelion_velocity =
        0.03029 / speed_of_light; // c
    static constexpr float_type mars_perihelion_velocity =
        0.0265 / speed_of_light; // c
    static constexpr float_type moon_average_velocity =
        1.022e-3 / speed_of_light; // c

    /// Convert the given distance in meters to the units of time in this system
    /// of units
    static constexpr float_type distance_from_si(float_type v) {
      return v * 1e-6; // Mm
    }

    /// Convert the given time in seconds to the units of time in this system of
    /// units
    static constexpr float_type time_from_si(float_type v) {
      return speed_of_light * v; // c^-1
    }

    /// Convert the given velocity in m/s to the units of time in this system of
    /// units
    static constexpr float_type velocity_from_si(float_type v) {
      return v * 1e-6 / speed_of_light; // c
    }
  };
} // namespace saga
