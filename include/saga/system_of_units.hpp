/*!\file Definition of different system of units

  All the systems of units have been adapted to make the speed of light be
  unity. This is chosen for convenience, to avoid multiplying it when
  calculating masses and velocities.

  Since the definition of distance, time and speed of light are linked
  together, maintaining the definition of the distance constant implies we
  have to change the definition of seconds, which must be provided in units
  of seconds / speed of light (thus multiplying the
  amount of seconds by the speed of light).
*/
#pragma once

namespace saga {

  /*!\brief System of units for to use in calculations at the Earth planetary
    system scale

    Values extracted from https://nssdc.gsfc.nasa.gov/planetary/factsheet

    Masses are given in thousands of tonnes. The distance is provided in km.
  */
  template <class TypeDescriptor> struct earth_system {

    using float_type = typename TypeDescriptor::float_type;

    static constexpr float_type speed_of_light = 299792.458; // km * s^-1
    static constexpr float_type gravitational_constant =
        7.4259154861063354e-25; // km * Gg^-1

    struct earth {
      static constexpr float_type mass = 5.9722e18; // Gg
    };

    struct moon {
      static constexpr float_type mass = 0.0123 * earth::mass; // Gg
      static constexpr float_type perigee = 363228.9;          // km
      static constexpr float_type perigee_velocity =
          1.082 / speed_of_light; // c
    };

    /// Convert the given time in seconds to the units of time in this system
    static constexpr float_type time_from_si(float_type v) {
      return speed_of_light * v; // c^-1
    }
  };

  /*!\brief System of units for to use in calculations at the solar system scale

    Values extracted from https://nssdc.gsfc.nasa.gov/planetary/factsheet

    Units are provided in the following way:

    * Masses referenced to the solar mass
    * The speed of light is set to unity
    * Distances are measured in thousands of kilometers
  */
  template <class TypeDescriptor> struct solar_system {

    using float_type = typename TypeDescriptor::float_type;

    // Constants
    static constexpr float_type gravitational_constant =
        1.47462e-6;                                           // Mo^-1 * Gm
    static constexpr float_type speed_of_light = 0.299792458; // Gm * s^-1

    struct sun {
      static constexpr float_type mass = 1.; // Mo
    };

    struct earth {
      static constexpr float_type mass = 3.003e-6 * sun::mass; // Mo
      static constexpr float_type perihelion = 147.10;         // Gm
      static constexpr float_type perihelion_velocity =
          30.29e-6 / speed_of_light; // c
    };

    struct mercury {
      static constexpr float_type mass = 0.0553 * earth::mass; // Mo
      static constexpr float_type perihelion = 46.002;         // Gm
      static constexpr float_type perihelion_velocity =
          69.8e-6 / speed_of_light; // c
    };

    struct venus {
      static constexpr float_type mass = 0.815 * earth::mass; // Mo
      static constexpr float_type perihelion = 107.476;       // Gm
      static constexpr float_type perihelion_velocity =
          35.26e-6 / speed_of_light; // c
    };

    // earth (defined above)

    struct mars {
      static constexpr float_type mass = 0.107 * earth::mass; // Mo
      static constexpr float_type perihelion = 206.7;         // Gm
      static constexpr float_type perihelion_velocity =
          26.5e-6 / speed_of_light; // c
    };

    struct jupiter {
      static constexpr float_type mass = 317.83 * earth::mass; // Mo
      static constexpr float_type perihelion = 740.522;        // Gm
      static constexpr float_type perihelion_velocity =
          13.72e-6 / speed_of_light; // c
    };

    struct saturn {
      static constexpr float_type mass = 95.16 * earth::mass; // Mo
      static constexpr float_type perihelion = 1352.555;      // Gm
      static constexpr float_type perihelion_velocity =
          10.18e-6 / speed_of_light; // c
    };

    struct uranus {
      static constexpr float_type mass = 14.54 * earth::mass; // Mo
      static constexpr float_type perihelion = 2741.302;      // Gm
      static constexpr float_type perihelion_velocity =
          7.11e-6 / speed_of_light; // c
    };

    struct neptune {
      static constexpr float_type mass = 17.15 * earth::mass; // Mo
      static constexpr float_type perihelion = 4444.449;      // Gm
      static constexpr float_type perihelion_velocity =
          5.50e-6 / speed_of_light; // c
    };

    /// Convert the given distance in meters to the units of time in this system
    static constexpr float_type distance_from_si(float_type v) {
      return v * 1e-9; // Gm
    }

    /// Convert the given time in seconds to the units of time in this system
    static constexpr float_type time_from_si(float_type v) {
      return speed_of_light * v; // c^-1
    }

    /// Convert the given velocity in m/s to the units of time in this system
    static constexpr float_type velocity_from_si(float_type v) {
      return v * 1e-6 / speed_of_light; // c
    }
  };
} // namespace saga
