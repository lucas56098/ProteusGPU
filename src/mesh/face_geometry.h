#pragma once

// the geometry of a face from what the mesh stores (included by mesh.h)

namespace mesh {

    HD inline Vec face_centroid_from_seed(const MeshPointers& mesh, int cell, int face) {
        const Vec to_neighbour = periodic_difference(mesh.seed[mesh.neighbour[face]], mesh.seed[cell]);
        return face_centroid_from_seed(mesh, face, to_neighbour, frame_around(to_neighbour));
    }

    // the middle of the two seeds, plus the centroid along the tangents of the frame
    HD inline Vec face_centroid_from_seed(const MeshPointers& mesh,
                                          int                 face,
                                          const Vec&          to_neighbour,
                                          const Frame&        frame) {
        Vec centroid = 0.5 * to_neighbour;
        for (int t = 0; t < config::dim - 1; t++)
            centroid += mesh.centroid_along_face[(config::dim - 1) * (long)face + t] * frame.tangent[t];
        return centroid;
    }

    HD inline double negligible_area_of(const MeshPointers& mesh, int cell) {
        double surface = 0.0;
        for (int f = mesh.first_face[cell]; f < mesh.first_face[cell] + mesh.n_faces[cell]; f++)
            surface += mesh.area[f];
        return NEGLIGIBLE_FACE * surface;
    }

} // namespace mesh
