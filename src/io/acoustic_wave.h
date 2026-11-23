#pragma once

// The acoustic wave of ACOUSTIC_TEST, made in the code instead of read: the cells of ics/create_acoustic_wave.py with
// its defaults and a jitter of 0.05, in copies side by side as tools/tile_ic.py of proteus_vs_arepo_timing puts them,
// number for number the same as these two files.

#include "config/config.h"
#include "math/math.h"

#include <cmath>
#include <cstdint>

namespace io {

    // a cell as a file of initial conditions holds it, with the total energy per volume
    struct CellOfFile {
        Vec    position;
        double density;
        Vec    velocity;
        double energy;
    };

    // a wave of unit length along x: rho_0 = 1, P_0 = 3/5 (c_s = 1 at gamma 5/3), delta_rho / rho_0 = 1e-6
    constexpr double WAVE_DENSITY   = 1.0;
    constexpr double WAVE_PRESSURE  = 3.0 / 5.0;
    constexpr double WAVE_AMPLITUDE = 1.0e-6;

    // every grid point moves by up to this share of a cell, by a hash of the cell and the seed
    constexpr double   WAVE_JITTER = 0.05;
    constexpr uint64_t WAVE_SEED   = 424242;

    // so that the rows of all cells fit a long
    constexpr long MOST_CELLS_PER_SIDE = 1L << 20;

    // uniform in [0, 1) from the cell and the axis alone, as per_particle_uniform of ics/common.py
    inline double uniform_of_cell(uint64_t cell, int axis) {
        uint64_t state = cell + WAVE_SEED * 0x9E3779B97F4A7C15ull + (uint64_t)axis * 0xC2B2AE3D27D4EB4Full;
        state ^= state >> 30;
        state *= 0xBF58476D1CE4E5B9ull;
        state ^= state >> 27;
        state *= 0x94D049BB133111EBull;
        state ^= state >> 31;
        return (double)(state >> 11) * (1.0 / (double)(1ull << 53));
    }

    // row of the waves_per_side^dim copies of the wave of cells_per_wave^dim cells, in the box [0, 1) of all of them;
    // the code units scale with the copies, so every copy has the steps of the wave alone
    inline CellOfFile acoustic_wave_cell(long row, long cells_per_wave, long waves_per_side) {
        const long n             = cells_per_wave;
        const long cells_of_wave = (config::dim == 2) ? n * n : n * n * n;
        const long copy          = row / cells_of_wave;
        const long cell          = row % cells_of_wave;

        // the grid of create_acoustic_wave.py; in 3D x runs over cell / n, y over cell / n^2, z over cell
        long column[3] = {0, 0, 0};
        if (config::dim == 2) {
            column[0] = cell % n;
            column[1] = cell / n;
        } else {
            column[0] = (cell / n) % n;
            column[1] = cell / (n * n);
            column[2] = cell % n;
        }
        const double dx = 1.0 / (double)n;
        Vec          x;
        for (int d = 0; d < config::dim; d++) {
            x[d] = ((double)column[d] + 0.5) * dx;
            x[d] = x[d] + (2.0 * uniform_of_cell((uint64_t)cell, d) - 1.0) * (WAVE_JITTER * dx);
        }

        const double gamma = config::adiabatic_index;
        const double sound = std::sqrt(gamma * WAVE_PRESSURE / WAVE_DENSITY);
        const double wave  = std::sin((2.0 * PI / 1.0) * x[0]);
        const double rho   = WAVE_DENSITY + WAVE_AMPLITUDE * wave;
        const double p     = WAVE_PRESSURE + sound * sound * WAVE_AMPLITUDE * wave;
        Vec          v{};
        v[0]            = (WAVE_AMPLITUDE / WAVE_DENSITY) * sound * wave;
        double v_square = 0.0;
        for (int d = 0; d < config::dim; d++)
            v_square = v_square + v[d] * v[d];
        const double energy = p / (gamma - 1.0) + 0.5 * rho * v_square;

        // the copy shifted to its place, lengths and speeds over waves_per_side, energies over its square
        const double k    = (double)waves_per_side;
        long         rest = copy;
        CellOfFile   result;
        for (int d = 0; d < config::dim; d++) {
            result.position[d] = (x[d] + (double)(rest % waves_per_side)) / k;
            result.velocity[d] = v[d] / k;
            rest /= waves_per_side;
        }
        result.density = rho;
        result.energy  = energy / (double)(waves_per_side * waves_per_side);
        return result;
    }

} // namespace io
