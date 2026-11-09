// implements the source terms (astro.h)

#include "astro/astro.h"

#include "logging/logging.h"
#include "mpi/mpi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace astro {

    namespace {

        // a cell that feeds the black hole
        struct ColdCell {
            long   id;
            double mass;
        };

        double cold_mass_summed_by_id(int n_cold, const int* cold_cell, const long* id, const double* cold_mass);
        void print_agn(double time, double cold_mass, const SourcePointers& of, const CodeUnits& units);
        HD double shortened_cfl_step(const Agn&            agn,
                                     const Vec&            position,
                                     const gas::Primitive& w,
                                     double                volume,
                                     const Vec&            mesh_velocity,
                                     double                cfl_fraction);
        HD void apply_first_half_to_cell(const SourcePointers& of, const Vec& position, double time, CellGas& gas);
        HD void apply_second_half_to_cell(const SourcePointers& of, const Vec& position, double time, CellGas& gas);

    } // namespace

    void Sources::set_up(const Parameters& parameters) {
        if constexpr (!config::astro_physics) return;
        units     = code_units_of(parameters.units);
        potential = potential_of(parameters, units);
        if constexpr (config::cooling) {
            read_cooling_table(parameters.cooling_table, cooling_table);
            cooling = cooling_of(parameters, units, cooling_table);
        }
        if constexpr (config::star_feedback) stars = stars_of(parameters, units);
        if constexpr (config::agn) agn = agn_of(parameters, units);
        if constexpr (config::limiters) limiters = limiters_of(parameters, units);
        logging::print("UNITS: 1 code unit = %g cm, %g g, %g cm/s",
                       parameters.units.length_in_cm,
                       parameters.units.mass_in_g,
                       parameters.units.velocity_in_cm_per_s);
        if constexpr (config::nfw || config::hernquist || config::smbh) {
            logging::print("GRAVITY: static potential enabled");
        }
        if constexpr (config::star_feedback) logging::print("STARS: SNIa + particle-free SF feedback enabled");
        if constexpr (config::agn) {
            char thermal[64] = "", kinetic[64] = "";
            if constexpr (config::agn_thermal) {
                std::snprintf(thermal, sizeof thermal, ", thermal f_T=%g", agn.thermal_share);
            }
            if constexpr (config::agn_kinetic) {
                std::snprintf(kinetic, sizeof kinetic, ", kinetic f_K=%g v_jet=%g", agn.kinetic_share, agn.jet_speed);
            }
            logging::print("AGN: cold-accretion trigger (R_acc=%g code)%s%s enabled",
                           std::sqrt(agn.accretion_radius_squared),
                           thermal,
                           kinetic);
        }
        if constexpr (config::limiters) {
            logging::print("LIMITERS: r<%g code, T<%g K, |v|<%g code enabled",
                           std::sqrt(limiters.radius_squared),
                           parameters.temperature_cap_in_k,
                           std::sqrt(limiters.speed_cap_squared));
        }
    }

    // the cold mass drains over the accretion time; a share of its rest energy comes back
    SourcePointers Sources::pointers() const {
        const double accretion_rate = cold_mass_near_the_centre / agn.accretion_time;
        const double power          = agn.efficiency * accretion_rate * agn.speed_of_light_squared;
        return SourcePointers{potential, cooling, stars, agn, limiters, accretion_rate, power};
    }

    double Sources::lowest_specific_energy() const {
        if constexpr (!config::cooling) return 0.0;
        return cooling.temperature_floor / cooling.kelvin_per_specific_energy;
    }

    void Sources::prepare(const timesteps::Sync&        sync,
                          const long*                   id,
                          const Vec*                    position,
                          const gas::PrimitivePointers& state,
                          const mesh::MeshPointers&     mesh) {
        if constexpr (!config::agn) return;
        cold_mass.grow_to_at_least(mesh.n_cells);
        double*   cold = cold_mass.data();
        const Agn agn  = this->agn;
        device::parallel_for(mesh.n_cells, [=] HD(long i) {
            const int     cell  = (int)i;
            const CellGas gas   = with_energy(gas::load(state, cell));
            const bool    feeds = feeds_the_black_hole(agn, from_centre(position[cell]), gas);
            cold[cell]          = feeds ? gas.density * mesh.volume[cell] : 0.0;
        });
        device::Array<int> cold_cell("astro: cells that feed the black hole", mesh.n_cells);
        const int          n_cold =
            device::indices_where(mesh.n_cells, [=] HD(int c) { return cold[c] > 0.0; }, cold_cell.data());
        cold_mass_near_the_centre = cold_mass_summed_by_id(n_cold, cold_cell.data(), id, cold);

        const bool fires = cold_mass_near_the_centre > 0.0;
        agn_switched     = fires && !agn_fired && !sync.ends_top_step;
        agn_fired        = fires;
        if (sync.ends_top_step) print_agn(sync.time, cold_mass_near_the_centre, pointers(), units);
    }

    void Sources::shorten_cfl_steps_near_the_agn(const mesh::CellList&         cells,
                                                 const Vec*                    position,
                                                 const gas::PrimitivePointers& state,
                                                 const Vec*                    mesh_velocity,
                                                 const mesh::MeshPointers&     mesh,
                                                 double                        cfl_fraction,
                                                 double*                       cfl_step) const {
        if constexpr (!config::agn) return;
        if (!agn_fired) return;
        const Agn agn = this->agn;
        device::parallel_for(cells.n, [=] HD(long i) {
            const int    cell    = cells[(int)i];
            const Vec    of_mesh = config::moving_mesh ? mesh_velocity[cell] : Vec{};
            const double step    = shortened_cfl_step(
                agn, position[cell], gas::load(state, cell), mesh.volume[cell], of_mesh, cfl_fraction);
            if (step < cfl_step[cell]) cfl_step[cell] = step;
        });
    }

    bool Sources::agn_switched_on() const {
        return agn_switched;
    }

    void Sources::apply_first_half(const timesteps::SourceTimes& time,
                                   const mesh::CellList&         active,
                                   const Vec*                    position,
                                   const gas::PrimitivePointers& state) const {
        if constexpr (!config::astro_physics) return;
        const SourcePointers of = pointers();
        device::parallel_for(active.n, [=] HD(long i) {
            const int cell = active[(int)i];
            CellGas   gas  = with_energy(gas::load(state, cell));
            apply_first_half_to_cell(of, position[cell], time[cell], gas);
            gas::store(state, cell, with_pressure(gas));
        });
    }

    void Sources::apply_second_half(const timesteps::SourceTimes& time,
                                    const mesh::CellList&         active,
                                    const Vec*                    position,
                                    const gas::PrimitivePointers& state) const {
        if constexpr (!config::astro_physics) return;
        const SourcePointers of = pointers();
        device::parallel_for(active.n, [=] HD(long i) {
            const int cell = active[(int)i];
            CellGas   gas  = with_energy(gas::load(state, cell));
            apply_second_half_to_cell(of, position[cell], time[cell], gas);
            gas::store(state, cell, with_pressure(gas));
        });
    }

    namespace {

        // the cold cells of all ranks go to every rank, so the sum is in id order on any number of ranks; collective
        double cold_mass_summed_by_id(int n_cold, const int* cold_cell, const long* id, const double* cold_mass) {
            std::vector<ColdCell> mine(n_cold);
            for (int i = 0; i < n_cold; i++)
                mine[i] = ColdCell{id[cold_cell[i]], cold_mass[cold_cell[i]]};
            const std::vector<char> bytes =
                mpi::gather_items_from_all_ranks(mine.data(), (int)mine.size(), (int)sizeof(ColdCell));
            std::vector<ColdCell> all(bytes.size() / sizeof(ColdCell));
            std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char*>(all.data()));
            std::sort(all.begin(), all.end(), [](const ColdCell& a, const ColdCell& b) { return a.id < b.id; });

            double sum = 0.0;
            for (const ColdCell& cell : all)
                sum += cell.mass;
            return sum;
        }

        void print_agn(double time, double cold_mass, const SourcePointers& of, const CodeUnits& units) {
            const double solar_masses = units.base.mass_in_g / SOLAR_MASS_IN_G;
            const double erg = units.base.mass_in_g * units.base.velocity_in_cm_per_s * units.base.velocity_in_cm_per_s;
            logging::print("AGN_POWER: t=%g Mcold_Msun=%g Mdot_Msun_per_yr=%g Edot_erg_per_s=%g",
                           time,
                           cold_mass * solar_masses,
                           of.accretion_rate * solar_masses * YEAR_IN_S / units.time_in_s,
                           of.power * erg / units.time_in_s);
        }

        // the sphere at the sound speed of the hottest heated gas, the jets at their speed cap
        HD double shortened_cfl_step(const Agn&            agn,
                                     const Vec&            position,
                                     const gas::Primitive& w,
                                     double                volume,
                                     const Vec&            mesh_velocity,
                                     double                cfl_fraction) {
            const Vec    offset = from_centre(position);
            const double sound  = in_the_heated_sphere(agn, offset) ? agn.sound_speed_of_hottest : 0.0;
            const double speed  = (config::agn_kinetic && in_a_jet(agn, offset)) ? agn.jet_speed_cap : 0.0;
            return hydro::cfl_step_of(w, volume, mesh_velocity, cfl_fraction, sound, speed);
        }

        HD void apply_first_half_to_cell(const SourcePointers& of, const Vec& position, double time, CellGas& gas) {
            if constexpr (config::static_potential) pull(of.potential, position, time, gas);
            if constexpr (config::cooling) cool(of.cooling, time, gas);
            if constexpr (config::star_feedback) let_stars_act(of.stars, position, time, gas);
            if constexpr (config::agn) {
                if (of.accretion_rate > 0.0)
                    feed_back(of.agn, position, dose_of(of.agn, of.accretion_rate, of.power, time), gas);
            }
        }

        HD void apply_second_half_to_cell(const SourcePointers& of, const Vec& position, double time, CellGas& gas) {
            if constexpr (config::agn) {
                if (of.accretion_rate > 0.0)
                    feed_back(of.agn, position, dose_of(of.agn, of.accretion_rate, of.power, time), gas);
            }
            if constexpr (config::star_feedback) let_stars_act(of.stars, position, time, gas);
            if constexpr (config::cooling) cool(of.cooling, time, gas);
            if constexpr (config::static_potential) pull(of.potential, position, time, gas);
            if constexpr (config::limiters) limit(of.limiters, position, gas);
        }

    } // namespace

} // namespace astro
