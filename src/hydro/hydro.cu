// implements the CFL step, the fluxes over the faces and the check for unphysical cells (hydro.h)

#include "hydro/hydro.h"

#include "logging/logging.h"
#include "mpi/mpi.h"

#include <cmath>

namespace hydro {

    namespace {

        // block shape of the flux kernel, tuned for the GH200
        constexpr int THREADS_PER_BLOCK             = 256;
        constexpr int MIN_BLOCKS_PER_MULTIPROCESSOR = 2;

        struct UnphysicalCells {
            long long density_not_positive;
            long long energy_not_positive;
            long long not_a_number;
        };

        HD double cfl_step_of_cell(int                           cell,
                                   const mesh::MeshPointers&     mesh,
                                   const gas::PrimitivePointers& state,
                                   const Vec*                    mesh_velocity,
                                   double                        cfl_fraction);
        HD void apply_fluxes_of_cell(int                                cell,
                                     const mesh::MeshPointers&          mesh,
                                     const gas::PrimitivePointers&      state,
                                     const gradients::GradientPointers& gradients,
                                     const Vec*                         mesh_velocity,
                                     const timesteps::StepPointers&     steps,
                                     timesteps::Half                    half,
                                     double                             lowest_specific_energy,
                                     const gas::ConservedPointers&      conserved);
        HD gas::Flux flux_over_face(int                                cell,
                                    int                                face,
                                    const gas::Primitive&              w,
                                    const gradients::Gradient&         gradient,
                                    const timesteps::FaceShare&        share,
                                    const mesh::MeshPointers&          mesh,
                                    const gas::PrimitivePointers&      state,
                                    const gradients::GradientPointers& gradients,
                                    const Vec*                         mesh_velocity,
                                    double                             lowest_specific_energy);
        HD Vec velocity_of_face(int                       face,
                                const Vec&                mesh_velocity_of_cell,
                                const Vec&                mesh_velocity_of_neighbour,
                                const Vec&                to_neighbour,
                                const Frame&              frame,
                                const mesh::MeshPointers& mesh);
        HD UnphysicalCells unphysical_cells_of(const gas::Primitive& w, double energy);

    } // namespace

    void cfl_steps(const mesh::MeshPointers&     mesh,
                   const gas::PrimitivePointers& state,
                   const Vec*                    mesh_velocity,
                   double                        cfl_fraction,
                   const mesh::CellList&         cells,
                   double*                       cfl_step) {
        device::parallel_for(cells.n, [=] HD(long i) {
            const int cell = cells[(int)i];
            cfl_step[cell] = cfl_step_of_cell(cell, mesh, state, mesh_velocity, cfl_fraction);
        });
    }

    void conserved_from_primitive(const mesh::MeshPointers&     mesh,
                                  const gas::PrimitivePointers& state,
                                  const mesh::CellList&         cells,
                                  const gas::ConservedPointers& conserved) {
        device::parallel_for(cells.n, [=] HD(long i) {
            const int cell = cells[(int)i];
            gas::store(conserved, cell, gas::conserved_of(gas::load(state, cell), mesh.volume[cell]));
        });
    }

    void primitive_from_conserved(const mesh::MeshPointers&     mesh,
                                  const gas::ConservedPointers& conserved,
                                  const mesh::CellList&         cells,
                                  double                        lowest_specific_energy,
                                  const gas::PrimitivePointers& state) {
        device::parallel_for(cells.n, [=] HD(long i) {
            const int      cell = cells[(int)i];
            gas::Primitive w    = gas::primitive_of(gas::load(conserved, cell), mesh.volume[cell]);
            if (lowest_specific_energy > 0.0)
                w = gas::with_temperature_floor(w, state.velocity[cell], lowest_specific_energy);
            gas::store(state, cell, w);
        });
    }

    void apply_fluxes(const mesh::MeshPointers&          mesh,
                      const gas::PrimitivePointers&      state,
                      const gradients::GradientPointers& gradients,
                      const Vec*                         mesh_velocity,
                      const timesteps::StepPointers&     steps,
                      timesteps::Half                    half,
                      double                             lowest_specific_energy,
                      const mesh::CellList&              cells,
                      const gas::ConservedPointers&      conserved) {
        device::parallel_for<THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR>(cells.n, [=] HD(long i) {
            apply_fluxes_of_cell(
                cells[(int)i], mesh, state, gradients, mesh_velocity, steps, half, lowest_specific_energy, conserved);
        });
    }

    void stop_if_unphysical(const gas::PrimitivePointers& state,
                            const gas::ConservedPointers& conserved,
                            const mesh::CellList&         cells) {
        const UnphysicalCells here = device::reduce(
            cells.n,
            UnphysicalCells{0, 0, 0},
            [=] HD(long i) {
                const int cell = cells[(int)i];
                return unphysical_cells_of(gas::load(state, cell), conserved.energy[cell]);
            },
            [] HD(const UnphysicalCells& a, const UnphysicalCells& b) {
                return UnphysicalCells{a.density_not_positive + b.density_not_positive,
                                       a.energy_not_positive + b.energy_not_positive,
                                       a.not_a_number + b.not_a_number};
            });

        long long count[3] = {here.density_not_positive, here.energy_not_positive, here.not_a_number};
        mpi::sum_over_ranks(count, 3);
        if (count[0] + count[1] + count[2] == 0) return;
        logging::stop_run(
            "HYDRO: unphysical cells: %lld with density <= 0, %lld with energy <= 0, %lld with a NaN; the run stops",
            count[0],
            count[1],
            count[2]);
    }

    namespace {

        HD double cfl_step_of_cell(int                           cell,
                                   const mesh::MeshPointers&     mesh,
                                   const gas::PrimitivePointers& state,
                                   const Vec*                    mesh_velocity,
                                   double                        cfl_fraction) {
            const Vec of_mesh = config::moving_mesh ? mesh_velocity[cell] : Vec{};
            return cfl_step_of(gas::load(state, cell), mesh.volume[cell], of_mesh, cfl_fraction, 0.0, 0.0);
        }

        // sums what flows out over the faces with a share, in face order; then the cell loses that
        HD void apply_fluxes_of_cell(int                                cell,
                                     const mesh::MeshPointers&          mesh,
                                     const gas::PrimitivePointers&      state,
                                     const gradients::GradientPointers& gradients,
                                     const Vec*                         mesh_velocity,
                                     const timesteps::StepPointers&     steps,
                                     timesteps::Half                    half,
                                     double                             lowest_specific_energy,
                                     const gas::ConservedPointers&      conserved) {
            const gas::Primitive      w        = gas::load(state, cell);
            const gradients::Gradient gradient = gradients::load(gradients, cell);

            gas::Conserved outflow{};
            const int      first = mesh.first_face[cell];
            for (int face = first; face < first + mesh.n_faces[cell]; face++) {
                const timesteps::FaceShare share = timesteps::face_share(steps, cell, mesh.neighbour[face], half);
                if (share.time == 0.0) continue;

                const gas::Flux flux = flux_over_face(
                    cell, face, w, gradient, share, mesh, state, gradients, mesh_velocity, lowest_specific_energy);
                const double amount = share.time * mesh.area[face];
                outflow.mass += amount * flux.mass;
                outflow.momentum += amount * flux.momentum;
                outflow.energy += amount * flux.energy;
            }

            gas::Conserved u = gas::load(conserved, cell);
            u.mass -= outflow.mass;
            u.momentum -= outflow.momentum;
            u.energy -= outflow.energy;
            gas::store(conserved, cell, u);
        }

        // HLLC flux over one face, in the frame of the box; the normal points from the cell to the neighbour
        HD gas::Flux flux_over_face(int                                cell,
                                    int                                face,
                                    const gas::Primitive&              w,
                                    const gradients::Gradient&         gradient,
                                    const timesteps::FaceShare&        share,
                                    const mesh::MeshPointers&          mesh,
                                    const gas::PrimitivePointers&      state,
                                    const gradients::GradientPointers& gradients,
                                    const Vec*                         mesh_velocity,
                                    double                             lowest_specific_energy) {
            const int   neighbour    = mesh.neighbour[face];
            const Vec   to_neighbour = periodic_difference(mesh.seed[neighbour], mesh.seed[cell]);
            const Frame frame        = frame_around(to_neighbour);
            const Vec   centroid     = mesh::face_centroid_from_seed(mesh, face, to_neighbour, frame);

            // on a moving mesh each side goes from where its seed was at the start of its step, and along the gas
            const gradients::Gradient gradient_of_neighbour    = gradients::load(gradients, neighbour);
            const gas::Primitive      w_of_neighbour           = gas::load(state, neighbour);
            Vec                       from_anchor              = centroid - gradient.anchor;
            Vec                       from_anchor_of_neighbour = centroid - to_neighbour - gradient_of_neighbour.anchor;
            if constexpr (config::moving_mesh) {
                from_anchor += share.since_start_of_cell * (mesh_velocity[cell] - w.velocity);
                from_anchor_of_neighbour +=
                    share.since_start_of_neighbour * (mesh_velocity[neighbour] - w_of_neighbour.velocity);
            }
            gas::Primitive left  = gradients::predicted(w, gradient, from_anchor, share.since_start_of_cell);
            gas::Primitive right = gradients::predicted(
                w_of_neighbour, gradient_of_neighbour, from_anchor_of_neighbour, share.since_start_of_neighbour);

            Vec face_velocity{};
            if constexpr (config::moving_mesh) {
                face_velocity =
                    velocity_of_face(face, mesh_velocity[cell], mesh_velocity[neighbour], to_neighbour, frame, mesh);
                left  = gas::in_moving_frame(left, face_velocity);
                right = gas::in_moving_frame(right, face_velocity);
            }

            const gas::Primitive left_in_frame  = gas::in_frame(gas::with_floors(left, lowest_specific_energy), frame);
            const gas::Primitive right_in_frame = gas::in_frame(gas::with_floors(right, lowest_specific_energy), frame);
            const gas::Flux      flux           = riemann::hllc(left_in_frame, right_in_frame);
            if constexpr (config::moving_mesh) {
                return gas::out_of_moving_frame(gas::out_of_frame(flux, frame), face_velocity);
            } else {
                return gas::out_of_frame(flux, frame);
            }
        }

        // mean of both mesh velocities plus the turn of the face about the seeds' middle, held back for a far centroid
        HD Vec velocity_of_face(int                       face,
                                const Vec&                mesh_velocity_of_cell,
                                const Vec&                mesh_velocity_of_neighbour,
                                const Vec&                to_neighbour,
                                const Frame&              frame,
                                const mesh::MeshPointers& mesh) {
            Vec from_middle{};
            for (int t = 0; t < config::dim - 1; t++)
                from_middle += mesh.centroid_along_face[(config::dim - 1) * (long)face + t] * frame.tangent[t];

            const double distance = sqrt(norm2(to_neighbour));
            const double off      = sqrt(norm2(from_middle));
            double       turn     = dot(from_middle, mesh_velocity_of_cell - mesh_velocity_of_neighbour) / distance;
            if (off > 0.9 * distance) turn *= (0.9 * distance) / off;
            return 0.5 * (mesh_velocity_of_cell + mesh_velocity_of_neighbour) + turn * frame.normal;
        }

        // a NaN is the one number that differs from itself
        HD UnphysicalCells unphysical_cells_of(const gas::Primitive& w, double energy) {
            bool not_a_number = (w.density != w.density) || (w.pressure != w.pressure) || (energy != energy);
            for (int d = 0; d < config::dim; d++)
                not_a_number = not_a_number || (w.velocity[d] != w.velocity[d]);
            return UnphysicalCells{w.density <= 0.0 ? 1 : 0, energy <= 0.0 ? 1 : 0, not_a_number ? 1 : 0};
        }

    } // namespace

} // namespace hydro
