// main simulation routine

#include "astro/sources.h"
#include "begrun/begrun.h"
#include "global/allvars.h"
#include "gradients/gradients.h"
#include "hydro/finite_volume_solver.h"
#include "io/output.h"
#include "mpi/decomp.h"
#include "mpi/halo.h"
#include "mpi/migrate.h"
#include "mpi/mpi_compat.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

/*=========================================================================
        _____           _                    _____ _____  _    _
       |  __ \         | |                  / ____|  __ \| |  | |
       | |__) | __ ___ | |_ ___ _   _ ___  | |  __| |__) | |  | |
       |  ___/ '__/ _ \| __/ _ \ | | / __| | | |_ |  ___/| |  | |
       | |   | | | (_) | ||  __/ |_| \__ \ | |__| | |    | |__| |
       |_|   |_|  \___/ \__\___|\__,_|___/  \_____|_|     \____/

       GPU-accelerated moving mesh hydrodynamics for astrophysics
===========================================================================
Version: 0.8
Authors: Lucas Schleuss, Dylan Nelson
Institution: Institute of Theoretical Astrophysics, Heidelberg University
===========================================================================*/

int main(int argc, char* argv[]) {

    // start MPI and set GPU of this rank
    mpi::init(&argc, &argv);

    // read params, load IC, first mesh, first snapshot
    begrun::begrun(argc, argv);

    {
        PROFILE("HYDRO");
        while (sim.t_sim < sim.t_end) {

            // prepare source terms
            astro::sources_prepare();

            // CFL timestep calculation
            const double dt = hydro::calc_timestep(sim.CFL, sim.mesh, sim.primvar);
            print_log();

            // sources first half
            astro::apply_sources_first_half(0.5 * dt);

            // gradients, mesh velocities
            mpi::exchange(sim.primvar);
            gradients::compute_prim_gradients(sim.mesh, sim.primvar, sim.grads);
#ifdef MOVING_MESH
            voronoi::compute_mesh_velocities(sim.mesh, sim.primvar, sim.grads);
#endif
            mpi::exchange(sim.grads, sim.mesh);

            // hydro first half
            hydro::prim_to_cons(sim.mesh, sim.primvar, sim.cons);
            hydro::apply_flux_update(0.5 * dt, 0.0, sim.mesh, sim.primvar, sim.grads, sim.cons);

#ifdef MOVING_MESH
            // move the seeds
            voronoi::move_seeds(sim.mesh, dt);

            // optional rebalance + migration of cells
            mpi::rebalance(sim.step, sim.mesh);
            mpi::migrate_cells(sim.mesh, sim.primvar, sim.cons, sim.grads);

            // build new mesh
            voronoi::compute_periodic_mesh(sim.mesh, sim.primvar, sim.cons, sim.grads);
            mpi::exchange(sim.primvar, sim.grads, sim.mesh);
#endif

            // hydro second half
            hydro::apply_flux_update(0.5 * dt, dt, sim.mesh, sim.primvar, sim.grads, sim.cons);
            hydro::cons_to_prim(sim.mesh, sim.cons, sim.primvar);

            // sources second half
            astro::apply_sources_second_half(0.5 * dt);

            // sanity check
            hydro::check_unphysical_state(sim.mesh, sim.primvar);

            sim.t_sim += dt;
            if (sim.t_sim >= sim.t_nextoutput || sim.t_sim >= sim.t_end) { output.write_snapshot(); }
            Profiler::log_timestep(sim.step);
            sim.step++;
        }
    }

    // free everything and print the summary
    begrun::endrun();
    mpi::finalize();
    return 0;
}
