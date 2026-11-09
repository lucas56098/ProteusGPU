// mesh velocities (mesh_motion.h): by distance as Springel 2010, by face angle as Vogelsberger et al. 2012

#include "mesh_motion/mesh_motion.h"

#include <cmath>

namespace mesh_motion {

    namespace {

        HD Vec mesh_velocity_of(int                           cell,
                                const mesh::MeshPointers&     mesh,
                                const gas::PrimitivePointers& state,
                                const Vec*                    density_gradient);
        HD Vec offset_of_seed(const Vec&            centroid_from_seed,
                              const gas::Primitive& w,
                              const Vec&            gradient,
                              double                radius);
        HD Vec regularization_by_distance(const Vec& offset, double radius, const gas::Primitive& w);
        HD Vec regularization_by_face_angle(int cell, const mesh::MeshPointers& mesh, const gas::Primitive& w);
        HD Vec regularization_by_volume(int cell, const mesh::MeshPointers& mesh, const gas::Primitive& w);

    } // namespace

    void mesh_velocities(const mesh::MeshPointers&     mesh,
                         const gas::PrimitivePointers& state,
                         const Vec*                    density_gradient,
                         const mesh::CellList&         cells,
                         Vec*                          mesh_velocity) {
        device::parallel_for(cells.n, [=] HD(long i) {
            const int cell      = cells[(int)i];
            mesh_velocity[cell] = mesh_velocity_of(cell, mesh, state, density_gradient);
        });
    }

    namespace {

        HD Vec mesh_velocity_of(int                           cell,
                                const mesh::MeshPointers&     mesh,
                                const gas::PrimitivePointers& state,
                                const Vec*                    density_gradient) {
            const gas::Primitive w = gas::load(state, cell);
            Vec                  regularization{};
            if constexpr (config::regularization_speed_by_distance > 0.0) {
                const double radius = radius_of_ball(mesh.volume[cell]);
                const Vec    offset = offset_of_seed(mesh.centroid_from_seed[cell], w, density_gradient[cell], radius);
                regularization      = regularization_by_distance(offset, radius, w);
            }
            if constexpr (config::regularization_speed_by_face_angle > 0.0) {
                const Vec by_face_angle = regularization_by_face_angle(cell, mesh, w);
                if (norm2(by_face_angle) > norm2(regularization)) regularization = by_face_angle;
            }
            if constexpr (config::regularization_speed_by_volume > 0.0) {
                regularization = regularization + regularization_by_volume(cell, mesh, w);
            }
            return w.velocity + regularization;
        }

        // seed to centroid, plus a step up the density gradient: smaller root of 2 x^2 - (3 R + L) x + R^2 = 0
        HD Vec offset_of_seed(const Vec&            centroid_from_seed,
                              const gas::Primitive& w,
                              const Vec&            gradient,
                              double                radius) {
            const double steepness = sqrt(norm2(gradient));
            if (!(steepness > 0.0 && radius > 0.0)) return centroid_from_seed;

            const double length             = w.density / steepness;
            const double linear_coefficient = 3.0 * radius + length;
            const double discriminant       = linear_coefficient * linear_coefficient - 8.0 * radius * radius;
            if (!(discriminant > 0.0)) return centroid_from_seed;

            const double step = fmin((linear_coefficient - sqrt(discriminant)) / 4.0, 0.25 * radius);
            return centroid_from_seed + (step * gradient) / steepness;
        }

        // nothing up to three quarters of the limit, full strength beyond it and linear in between, at the sound speed
        HD Vec regularization_by_distance(const Vec& offset, double radius, const gas::Primitive& w) {
            const double distance = sqrt(norm2(offset));
            const double limit    = OFF_CENTRE_LIMIT * radius;
            if (!(radius > 0.0 && distance > 0.75 * limit) || !(w.density > 0.0 && w.pressure > 0.0)) return Vec{};

            const double full     = config::regularization_speed_by_distance;
            const double strength = (distance > limit) ? full : full * (distance - 0.75 * limit) / (0.25 * limit);
            return ((strength * gas::sound_speed(w)) * offset) / distance;
        }

        // same ramp as by distance, on the largest face angle of the cell, toward the centroid
        HD Vec regularization_by_face_angle(int cell, const mesh::MeshPointers& mesh, const gas::Primitive& w) {
            double largest = 0.0;
            for (int f = mesh.first_face[cell]; f < mesh.first_face[cell] + mesh.n_faces[cell]; f++) {
                const double half_way =
                    0.5 * sqrt(norm2(periodic_difference(mesh.seed[mesh.neighbour[f]], mesh.seed[cell])));
                if (!(half_way > 0.0)) continue;
                // half the face in 2D, the radius of a disc of its area in 3D, against half the way to the neighbour
                const double angle =
                    (config::dim == 2) ? 0.5 * mesh.area[f] / half_way : sqrt(mesh.area[f] / M_PI) / half_way;
                largest = fmax(largest, angle);
            }

            const Vec    offset   = mesh.centroid_from_seed[cell];
            const double distance = sqrt(norm2(offset));
            if (!(largest > 0.75 * LARGEST_FACE_ANGLE && distance > 0.0) || !(w.density > 0.0 && w.pressure > 0.0))
                return Vec{};

            const double full     = config::regularization_speed_by_face_angle;
            const double strength = (largest > LARGEST_FACE_ANGLE)
                                        ? full
                                        : full * (largest - 0.75 * LARGEST_FACE_ANGLE) / (0.25 * LARGEST_FACE_ANGLE);
            return ((strength * gas::sound_speed(w)) * offset) / distance;
        }

        // a cell below the reference size moves towards its larger neighbours, at the faster of sound and gas speed
        HD Vec regularization_by_volume(int cell, const mesh::MeshPointers& mesh, const gas::Primitive& w) {
            const double radius = radius_of_ball(mesh.volume[cell]);
            if (!(radius > 0.0 && mesh.reference_radius > 0.0) || !(w.density > 0.0 && w.pressure > 0.0)) return Vec{};

            const double ratio = mesh.reference_radius / radius;
            if (!(ratio > LARGEST_RADIUS_RATIO)) return Vec{};

            Vec towards_larger{};
            for (int f = mesh.first_face[cell]; f < mesh.first_face[cell] + mesh.n_faces[cell]; f++) {
                const Vec    to_neighbour = periodic_difference(mesh.seed[mesh.neighbour[f]], mesh.seed[cell]);
                const double distance     = sqrt(norm2(to_neighbour));
                if (!(distance > 0.0)) continue;
                const double weight = mesh.area[f] * (mesh.volume[mesh.neighbour[f]] - mesh.volume[cell]) / distance;
                towards_larger      = towards_larger + weight * to_neighbour;
            }
            const double length = sqrt(norm2(towards_larger));
            if (!(length > 0.0)) return Vec{};

            const double strength = config::regularization_speed_by_volume *
                                    fmin((ratio - LARGEST_RADIUS_RATIO) / LARGEST_RADIUS_RATIO, 1.0);
            const double speed = fmax(gas::sound_speed(w), sqrt(norm2(w.velocity)));
            return ((strength * speed) * towards_larger) / length;
        }

    } // namespace

} // namespace mesh_motion
