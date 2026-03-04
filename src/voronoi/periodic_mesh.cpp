#include "periodic_mesh.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <climits>
#include "global/allvars.h"
#include "io/input.h"
#include "io/output.h"
#include "knn/knn.h"
#include "begrun/begrun.h"
#include "voronoi/voronoi.h"

namespace voronoi {

inline bool is_in(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za, double zb) {
    #ifdef dim_2D
    return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb);
    #else
    return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb) && (pt.z > za && pt.z < zb);
    #endif
}

inline void add_ghost(POINT_TYPE* pts, hsize_t index, hsize_t* n_ghosts,  const hsize_t* n_hydro, hsize_t* original_ids, double shift_x, double shift_y, double shift_z) {
    // create shifted pt
    POINT_TYPE pt;
    pt.x = pts[index].x + shift_x;
    pt.y = pts[index].y + shift_y;
    #ifdef dim_3D
    pt.z = pts[index].z + shift_z;
    #else
    (void)shift_z;
    #endif

    // add pt to pts
    pts[(*n_hydro) + (*n_ghosts)] = pt;
    original_ids[*n_ghosts] = index;
    (*n_ghosts)++;
}

// for now only 2D
VMesh* compute_periodic_mesh(POINT_TYPE* pts_data, hsize_t num_points) {

    std::cout << "VORONOI: set up periodic mesh" << std::endl;

    // allocate new pts (that include ghosts)
    hsize_t max_ghost_points = (DIMENSION+1) * num_points; // naive guess, will be shortened later
    POINT_TYPE* pts;
    pts = (POINT_TYPE*)malloc((num_points + max_ghost_points) * sizeof(POINT_TYPE));

    hsize_t n_ghosts = 0;
    hsize_t n_hydro = num_points;
    hsize_t* original_ids;
    original_ids = (hsize_t*)malloc(max_ghost_points * sizeof(hsize_t)); // naive guess, will be shortened later

    #ifdef DEBUG_MODE
    std::cout << "VORONOI: select which ghostcells are needed" << std::endl;
    #endif

    // select points that get ghosts
    for (hsize_t i = 0; i < n_hydro; i++) {
        
        // copy original point to pts
        pts[i] = pts_data[i];

        #ifdef dim_2D
        // check if point is in any of those regions... if so add the corresponding ghost
        // edges
        if (is_in(pts[i], 0., 0.+buff, 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 0.);} // region 1
        if (is_in(pts[i], 0., 1., 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., -1.);} // region 2
        if (is_in(pts[i], 1.-buff, 1., 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 0.);} // region 3
        if (is_in(pts[i], 0., 1., 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 1.);} // region 4
        // corners
        if (is_in(pts[i], 0., buff, 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 1.);} // region 5
        if (is_in(pts[i], 0., buff, 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., -1.);} // region 6
        if (is_in(pts[i], 1.-buff, 1., 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., -1.);} // region 7
        if (is_in(pts[i], 1.-buff, 1., 0, buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 1.);} // region 8
        #else
        // check if point is in any of those regions... if so add the corresponding ghost
        // faces
        if (is_in(pts[i], 0., 1., 0., 1., 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 0., 1.);} // 1
        if (is_in(pts[i], 0., buff, 0., 1., 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 0., 0.);} // 2
        if (is_in(pts[i], 0., 1., 1.-buff, 1., 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., -1., 0.);} // 3
        if (is_in(pts[i], 1.-buff, 1., 0., 1., 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 0., 0.);} // 4
        if (is_in(pts[i], 0., 1., 0., buff, 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 1., 0.);} // 5
        if (is_in(pts[i], 0., 1., 0., 1., 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 0., -1.);} // 6
        // edges
        if (is_in(pts[i], 0., 1., 0., buff, 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 1., 1.);} // 1
        if (is_in(pts[i], 0., buff, 0., 1, 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 0., 1.);} // 2
        if (is_in(pts[i], 0., 1., 1.-buff, 1., 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., -1., 1.);} // 3
        if (is_in(pts[i], 1.-buff, 1., 0., 1., 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 0., 1.);} // 4
        if (is_in(pts[i], 0., buff, 0., buff, 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 1., 0.);} // 5
        if (is_in(pts[i], 0., buff, 1.-buff, 1., 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., -1., 0.);} // 6
        if (is_in(pts[i], 1.-buff, 1., 1.-buff, 1., 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., -1., 0.);} // 7
        if (is_in(pts[i], 1.-buff, 1., 0., buff, 0., 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 1., 0.);} // 8
        if (is_in(pts[i], 0., buff, 0., 1., 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 0., -1.);} // 9
        if (is_in(pts[i], 0., 1., 1.-buff, 1., 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., -1., -1.);} // 10
        if (is_in(pts[i], 1.-buff, 1., 0., 1., 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 0., -1.);} // 11
        if (is_in(pts[i], 0., 1., 0., buff, 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 1., -1.);} // 12
        // corners
        if (is_in(pts[i], 0., buff, 0., buff, 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 1., 1.);} // 1
        if (is_in(pts[i], 0., buff, 1.-buff, 1., 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., -1., 1.);} // 2
        if (is_in(pts[i], 1.-buff, 1., 1.-buff, 1., 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., -1., 1.);} // 3
        if (is_in(pts[i], 1.-buff, 1., 0., buff, 0., buff)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 1., 1.);} // 4
        if (is_in(pts[i], 0., buff, 0., buff, 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 1., -1.);} // 5
        if (is_in(pts[i], 0., buff, 1.-buff, 1., 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., -1., -1.);} // 6
        if (is_in(pts[i], 1.-buff, 1., 1.-buff, 1., 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., -1., -1.);} // 7
        if (is_in(pts[i], 1.-buff, 1., 0., buff, 1.-buff, 1.)) {add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 1., -1.);} // 8
        #endif
    }

    #ifdef DEBUG_MODE
    std::cout << "VORONOI: scale mesh down to [0,1]^d" << std::endl;
    #endif

    // scale down... to [0,1]^2
    double scale = 1./(1.+(2*buff));
    for (hsize_t i = 0; i < n_hydro + n_ghosts; i++) {
        pts[i].x = scale * (pts[i].x - 0.5) + 0.5;
        pts[i].y = scale * (pts[i].y - 0.5) + 0.5;
        #ifdef dim_3D
        pts[i].z = scale * (pts[i].z - 0.5) + 0.5;
        #endif
    }

    // compute mesh
    pts = (POINT_TYPE*)realloc(pts, (n_hydro + n_ghosts)*sizeof(POINT_TYPE));
    VMesh* mesh = compute_mesh(pts, n_hydro + n_ghosts);
    free(pts);

    // set mesh ghost quantities
    #ifdef DEBUG_MODE
    std::cout << "VORONOI: copy ghost quantities to VMesh" << std::endl;
    #endif
    mesh->n_hydro = n_hydro;
    mesh->ghost_ids = (hsize_t*)realloc(original_ids, n_ghosts*sizeof(hsize_t));

    
    // scale mesh up
    #ifdef DEBUG_MODE
    std::cout << "VORONOI: scale mesh back to [-buff, 1+buff]^d" << std::endl;
    #endif

    scale = 1. + (2*buff);
    #ifdef dim_2D
    double vscale = scale*scale;
    double ascale = scale;
    #else
    double vscale = scale*scale*scale;
    double ascale = scale*scale;
    #endif

    for (hsize_t i = 0; i < n_hydro + n_ghosts; i++) {
        mesh->seeds[i].x = (mesh->seeds[i].x - 0.5) * scale + 0.5;
        mesh->seeds[i].y = (mesh->seeds[i].y - 0.5) * scale + 0.5;
        #ifdef dim_3D
        mesh->seeds[i].z = (mesh->seeds[i].z - 0.5) * scale + 0.5;
        #endif
        mesh->volumes[i] = vscale*mesh->volumes[i];
    }

    for (hsize_t i = 0; i < mesh->num_faces; i++) {
        mesh->face_area[i] = ascale*mesh->face_area[i];
    }

    #ifdef DEBUG_MODE
    for (hsize_t i = 0; i < mesh->num_edge_coord_verts; i++) {
        mesh->edge_coords[DIMENSION*i] = (mesh->edge_coords[DIMENSION*i] - 0.5) * scale + 0.5;
        mesh->edge_coords[DIMENSION*i + 1] = (mesh->edge_coords[DIMENSION*i + 1] - 0.5) * scale + 0.5;
        #ifdef dim_3D
        mesh->edge_coords[DIMENSION*i + 2] = (mesh->edge_coords[DIMENSION*i + 2] - 0.5) * scale + 0.5;
        #endif
    }
    #endif

    #ifdef DEBUG_MODE
    std::cout << "VORONOI: periodic mesh should be created" << std::endl;
    #endif

    // return that mesh :D
    return mesh;
}

} // namespace voronoi