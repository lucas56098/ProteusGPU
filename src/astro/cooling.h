#pragma once

// Radiative cooling from a table, integrated exactly over a step (Townsend 2009), down to a temperature floor.

#include "astro/gas_of_cell.h"
#include "astro/parameters.h"
#include "device/device.h"

#include <cmath>
#include <string>

namespace astro {

    // rows by rising temperature, the rate a power law between them; Y is the scaled cooling time from the hottest row
    struct CoolingTable {
        int                   n_rows = 0;
        device::Array<double> temperature{"astro: temperatures of the cooling table", 0};
        device::Array<double> rate{"astro: cooling rates of the table", 0};
        device::Array<double> slope{"astro: power law slopes between the rows", 0};
        device::Array<double> scaled_time{"astro: scaled cooling times of the rows", 0};
    };

    // what a kernel reads
    struct Cooling {
        int           n_rows = 0;
        const double* temperature;
        const double* rate;
        const double* slope;
        const double* scaled_time;
        double        hottest_temperature;
        double        hottest_rate;
        double        temperature_floor;
        double        kelvin_per_specific_energy;
        double        scaled_time_per_density_and_time;
    };

    // reads rows of temperature in K and rate in erg cm^3 / s; no file or fewer than two rows stops the run
    void read_cooling_table(const std::string& path, CoolingTable& table);

    inline Cooling cooling_of(const Parameters& parameters, const CodeUnits& units, const CoolingTable& table) {
        Cooling cooling;
        cooling.n_rows                     = table.n_rows;
        cooling.temperature                = table.temperature.data();
        cooling.rate                       = table.rate.data();
        cooling.slope                      = table.slope.data();
        cooling.scaled_time                = table.scaled_time.data();
        cooling.hottest_temperature        = table.temperature[cooling.n_rows - 1];
        cooling.hottest_rate               = table.rate[cooling.n_rows - 1];
        cooling.temperature_floor          = parameters.temperature_floor_in_k;
        cooling.kelvin_per_specific_energy = units.kelvin_per_specific_energy;
        cooling.scaled_time_per_density_and_time =
            (cooling.hottest_rate / cooling.hottest_temperature) * (config::adiabatic_index - 1.0) *
            MEAN_MOLECULAR_WEIGHT * HYDROGEN_MASS_FRACTION * units.density_in_g_per_cm3 * units.time_in_s /
            (MASS_PER_ELECTRON * BOLTZMANN_CONSTANT * PROTON_MASS);
        return cooling;
    }

    HD inline int row_below(double T, const Cooling& c) {
        for (int k = 0; k < c.n_rows - 1; k++) {
            if (T < c.temperature[k + 1]) return k;
        }
        return c.n_rows - 2;
    }

    // the integral of the power law from the row up to the temperature; a slope of 1 makes it a logarithm
    HD inline double scaled_time_above_row(double T, int k, const Cooling& c) {
        const double a      = c.slope[k];
        const double ratio  = T / c.temperature[k];
        const double factor = (c.hottest_rate / c.hottest_temperature) * c.temperature[k] / c.rate[k];
        return (fabs(1.0 - a) > 1e-6) ? factor / (1.0 - a) * (power(ratio, 1.0 - a) - 1.0)
                                      : factor * natural_log(ratio);
    }

    HD inline double scaled_time_of(double T, const Cooling& c) {
        const int k = row_below(T, c);
        return c.scaled_time[k] - scaled_time_above_row(T, k, c);
    }

    // temperature of a Y; beyond the coolest row that row
    HD inline double temperature_of_scaled_time(double Y, const Cooling& c) {
        if (Y >= c.scaled_time[0]) return c.temperature[0];
        int k = c.n_rows - 2;
        for (int j = 0; j < c.n_rows - 1; j++) {
            if (Y >= c.scaled_time[j + 1]) {
                k = j;
                break;
            }
        }
        const double a = c.slope[k];
        const double x =
            (c.scaled_time[k] - Y) * (c.hottest_temperature * c.rate[k]) / (c.hottest_rate * c.temperature[k]);
        const double ratio = (fabs(1.0 - a) > 1e-6) ? power(1.0 + (1.0 - a) * x, 1.0 / (1.0 - a)) : exponential(x);
        return c.temperature[k] * ratio;
    }

    // a step in Y over the time, then back to a temperature, at least the floor; a cell at or below the floor stays
    HD inline void cool(const Cooling& c, double time, CellGas& gas) {
        const double internal = internal_energy(gas);
        if (internal <= 0.0) return;
        const double T = c.kelvin_per_specific_energy * (internal / gas.density);
        if (T <= c.temperature_floor) return;

        double T_new = temperature_of_scaled_time(
            scaled_time_of(T, c) + c.scaled_time_per_density_and_time * gas.density * time, c);
        if (T_new < c.temperature_floor) T_new = c.temperature_floor;
        gas.energy += gas.density * (T_new - T) / c.kelvin_per_specific_energy;
    }

} // namespace astro
