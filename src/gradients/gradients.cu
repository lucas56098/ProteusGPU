// implements the gradient estimate (gradients.h)

#include "gradients/gradients.h"

#include <cmath>

namespace gradients {

    namespace {

        // block shape of the kernel, tuned for the GH200
        constexpr int THREADS_PER_BLOCK             = 256;
        constexpr int MIN_BLOCKS_PER_MULTIPROCESSOR = 2;

        // a state as one row: density, the velocity components, pressure
        constexpr int N_VARIABLES = config::dim + 2;

        // a neighbour this close (squared distance) to the centroid would get an endless weight, so it is left out
        constexpr double TOO_CLOSE = 1e-24;

        // determinant over product of the diagonal below this: the matrix counts as singular and the gradient is zero
        constexpr double SINGULAR = 1e-12;

        // the normal equations of the fit: one matrix shared by all variables, one right side for each
        struct NormalEquations {
            double matrix[config::dim][config::dim] = {};
            Vec    right_side[N_VARIABLES]          = {};

            HD void add(double weight, const Vec& to_neighbour, const double* difference);
        };

        // lowest and highest value of every variable over the cell and its neighbours
        struct Range {
            double lowest[N_VARIABLES];
            double highest[N_VARIABLES];

            HD void start_at(const double* value);
            HD void include(const double* value);
        };

        HD void compute_gradient_of_cell(int                            cell,
                                         const mesh::MeshPointers&      mesh,
                                         const gas::PrimitivePointers&  state,
                                         const Vec*                     mesh_velocity,
                                         const timesteps::StepPointers& steps,
                                         const GradientPointers&        gradients);
        HD gas::Primitive state_at_sync(int                            other,
                                        const mesh::MeshPointers&      mesh,
                                        const gas::PrimitivePointers&  state,
                                        const Vec*                     mesh_velocity,
                                        const timesteps::StepPointers& steps,
                                        const GradientPointers&        gradients);
        HD void as_row(const gas::Primitive& w, double* row);
        HD void solve(const NormalEquations& equations, Vec* gradient);
        HD double limit_factor(double value, double lowest, double highest, double change);

    } // namespace

    void compute_gradients(const mesh::MeshPointers&      mesh,
                           const gas::PrimitivePointers&  state,
                           const Vec*                     mesh_velocity,
                           const timesteps::StepPointers& steps,
                           const mesh::CellList&          cells,
                           const GradientPointers&        gradients) {
        device::parallel_for<THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR>(cells.n, [=] HD(long i) {
            compute_gradient_of_cell(cells[(int)i], mesh, state, mesh_velocity, steps, gradients);
        });
    }

    namespace {

        // fit over the faces from the centroid, then every gradient scaled to keep the face centroids in range
        HD void compute_gradient_of_cell(int                            cell,
                                         const mesh::MeshPointers&      mesh,
                                         const gas::PrimitivePointers&  state,
                                         const Vec*                     mesh_velocity,
                                         const timesteps::StepPointers& steps,
                                         const GradientPointers&        gradients) {
            const Vec anchor = mesh.centroid_from_seed[cell];
            double    value[N_VARIABLES];
            as_row(gas::load(state, cell), value);

            NormalEquations equations;
            Range           range;
            range.start_at(value);

            const int    first      = mesh.first_face[cell];
            const int    end        = first + mesh.n_faces[cell];
            const double negligible = mesh::negligible_area_of(mesh, cell);
            for (int f = first; f < end; f++) {
                if (mesh.area[f] <= negligible) continue;
                const int other        = mesh.neighbour[f];
                const Vec other_anchor = mesh.centroid_from_seed[other];
                const Vec to_neighbour = periodic_difference(mesh.seed[other], mesh.seed[cell]) + other_anchor - anchor;
                const double squared_distance = norm2(to_neighbour);
                if (squared_distance < TOO_CLOSE) continue;

                double other_value[N_VARIABLES], difference[N_VARIABLES];
                as_row(state_at_sync(other, mesh, state, mesh_velocity, steps, gradients), other_value);
                for (int r = 0; r < N_VARIABLES; r++)
                    difference[r] = other_value[r] - value[r];
                equations.add(mesh.area[f] / squared_distance, to_neighbour, difference);
                range.include(other_value);
            }

            Vec gradient[N_VARIABLES];
            solve(equations, gradient);

            double factor[N_VARIABLES];
            for (int r = 0; r < N_VARIABLES; r++)
                factor[r] = 1.0;
            for (int f = first; f < end; f++) {
                if (mesh.area[f] <= negligible) continue;
                const Vec to_face = mesh::face_centroid_from_seed(mesh, cell, f) - anchor;
                for (int r = 0; r < N_VARIABLES; r++) {
                    const double change = dot(gradient[r], to_face);
                    factor[r] = fmin(factor[r], limit_factor(value[r], range.lowest[r], range.highest[r], change));
                }
            }

            Gradient limited;
            limited.density = factor[0] * gradient[0];
            for (int d = 0; d < config::dim; d++)
                limited.velocity[d] = factor[1 + d] * gradient[1 + d];
            limited.pressure = factor[config::dim + 1] * gradient[config::dim + 1];
            limited.anchor   = anchor;
            store(gradients, cell, limited);
        }

        // a neighbour in the middle of its step holds the state of its start, so it is predicted to the sync
        HD gas::Primitive state_at_sync(int                            other,
                                        const mesh::MeshPointers&      mesh,
                                        const gas::PrimitivePointers&  state,
                                        const Vec*                     mesh_velocity,
                                        const timesteps::StepPointers& steps,
                                        const GradientPointers&        gradients) {
            const gas::Primitive w = gas::load(state, other);
            if constexpr (!config::individual_timesteps) {
                return w;
            } else {
                const double since_start = timesteps::since_start(steps, other);
                if (since_start == 0.0) return w;
                const Gradient gradient = load(gradients, other);
                Vec            r        = mesh.centroid_from_seed[other] - gradient.anchor;
                if constexpr (config::moving_mesh) r += since_start * (mesh_velocity[other] - w.velocity);
                return predicted(w, gradient, r, since_start);
            }
        }

        HD void as_row(const gas::Primitive& w, double* row) {
            row[0] = w.density;
            for (int d = 0; d < config::dim; d++)
                row[1 + d] = w.velocity[d];
            row[config::dim + 1] = w.pressure;
        }

        // every product once, so that the matrix stays exactly symmetric
        HD void NormalEquations::add(double weight, const Vec& to_neighbour, const double* difference) {
            for (int a = 0; a < config::dim; a++) {
                for (int b = a; b < config::dim; b++) {
                    const double term = weight * to_neighbour[a] * to_neighbour[b];
                    matrix[a][b] += term;
                    if (b != a) matrix[b][a] += term;
                }
            }
            for (int r = 0; r < N_VARIABLES; r++)
                right_side[r] += (weight * difference[r]) * to_neighbour;
        }

        HD void Range::start_at(const double* value) {
            for (int r = 0; r < N_VARIABLES; r++) {
                lowest[r]  = value[r];
                highest[r] = value[r];
            }
        }

        HD void Range::include(const double* value) {
            for (int r = 0; r < N_VARIABLES; r++) {
                lowest[r]  = fmin(lowest[r], value[r]);
                highest[r] = fmax(highest[r], value[r]);
            }
        }

        // solves with the inverse of the symmetric matrix (Cramer's rule), the same for every variable
        HD void solve(const NormalEquations& equations, Vec* gradient) {
            // inverse is the inverse of the matrix times its determinant
            const auto& m = equations.matrix;
            double      inverse[config::dim][config::dim];
            double      determinant, diagonal;
            if constexpr (config::dim == 2) {
                determinant   = m[0][0] * m[1][1] - m[0][1] * m[1][0];
                diagonal      = m[0][0] * m[1][1];
                inverse[0][0] = m[1][1];
                inverse[0][1] = -m[0][1];
                inverse[1][0] = -m[1][0];
                inverse[1][1] = m[0][0];
            } else {
                inverse[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];
                inverse[0][1] = m[0][2] * m[2][1] - m[0][1] * m[2][2];
                inverse[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1];
                inverse[1][0] = m[1][2] * m[2][0] - m[1][0] * m[2][2];
                inverse[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];
                inverse[1][2] = m[0][2] * m[1][0] - m[0][0] * m[1][2];
                inverse[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0];
                inverse[2][1] = m[0][1] * m[2][0] - m[0][0] * m[2][1];
                inverse[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];
                determinant   = m[0][0] * inverse[0][0] + m[0][1] * inverse[1][0] + m[0][2] * inverse[2][0];
                diagonal      = m[0][0] * m[1][1] * m[2][2];
            }

            for (int r = 0; r < N_VARIABLES; r++)
                gradient[r] = Vec{};
            if (!(fabs(determinant) > SINGULAR * diagonal)) return;

            for (int r = 0; r < N_VARIABLES; r++) {
                for (int a = 0; a < config::dim; a++) {
                    double sum = 0.0;
                    for (int b = 0; b < config::dim; b++)
                        sum += inverse[a][b] * equations.right_side[r][b];
                    gradient[r][a] = sum / determinant;
                }
            }
        }

        // the factor that keeps value + change between lowest and highest
        HD double limit_factor(double value, double lowest, double highest, double change) {
            double factor = 1.0;
            if (change > 0.0 && value + change > highest) factor = (highest > value) ? (highest - value) / change : 0.0;
            if (change < 0.0 && value + change < lowest) factor = (lowest < value) ? (lowest - value) / change : 0.0;
            return fmin(fmax(factor, 0.0), 1.0);
        }

    } // namespace

} // namespace gradients
