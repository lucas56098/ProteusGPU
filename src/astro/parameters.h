#pragma once

// The parameters of the astrophysical sources as param.txt gives them.

#include "astro/units.h"

#include <string>

namespace astro {

    // the parameters of the switches turned on; params reads only those
    struct Parameters {
        Units units;

        // static potential
        double halo_mass_in_solar_masses       = 0.0; // M_NFW
        double halo_concentration              = 0.0; // c_NFW
        double hubble_constant_in_km_per_s_mpc = 0.0; // H0
        double galaxy_mass_in_solar_masses     = 0.0; // M_BCG, also of the stars
        double galaxy_radius_in_kpc            = 0.0; // R_BCG, also of the stars
        double black_hole_mass_in_solar_masses = 0.0; // M_BH
        double black_hole_softening_in_kpc     = 0.0; // smbh_softening

        // cooling
        std::string cooling_table; // cooling_table: rows of temperature in K and rate in erg cm^3 / s
        double      temperature_floor_in_k = 0.0; // T_floor

        // stars
        double supernova_rate_per_year_and_solar_mass = 0.0; // Gamma_SNIa
        double supernova_energy_in_erg                = 0.0; // E_SNIa
        double mass_loss_rate_per_s                   = 0.0; // alpha_SNIa
        double star_formation_efficiency              = 0.0; // eff_SF
        double star_formation_density_in_per_cm3      = 0.0; // n_SF, hydrogen atoms
        double star_formation_temperature_in_k        = 0.0; // T_SF
        double star_formation_radius_in_kpc           = 0.0; // R_SF

        // the black hole; the accretion radius is also the inner radius of the star formation
        double accretion_radius_in_kpc         = 0.0; // R_acc
        double cold_temperature_in_k           = 0.0; // T_cold_acc
        double accretion_time_in_myr           = 0.0; // t_acc
        double feedback_efficiency             = 0.0; // eta_agn
        double heated_radius_in_kpc            = 0.0; // R_T
        double thermal_share                   = 0.0; // f_T
        double kinetic_share                   = 0.0; // f_K
        double jet_radius_in_kpc               = 0.0; // R_jet
        double jet_height_in_kpc               = 0.0; // h_jet
        double jet_distance_in_kpc             = 0.0; // L_jet
        double jet_speed_cap_in_speed_of_light = 0.0; // v_cap

        // limiters
        double limited_radius_in_kpc       = 0.0; // R_lim
        double temperature_cap_in_k        = 0.0; // T_max_lim
        double speed_cap_in_speed_of_light = 0.0; // v_cap_lim
    };

} // namespace astro
