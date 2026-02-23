#include "voronoi.h"
#include "../global/allvars.h"
#include "../knn/knn.h"
#include "../io/input.h"
#include "../io/output.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

namespace voronoi {

    static uchar3 tr_data[_VORO_BLOCK_SIZE_ * _MAX_T_]; // memory pool for chained lists of triangles
    static uchar boundary_next_data[_VORO_BLOCK_SIZE_ * _MAX_P_];
    static double4 clip_data[_VORO_BLOCK_SIZE_ * _MAX_P_]; // clipping planes

    static const uchar END_OF_LIST = 255;

    #ifdef CPU_DEBUG
    static int blockId, threadId;
    static inline uchar3& tr(int t) { return tr_data[threadId * _MAX_T_ + t]; } // threadId.x for gpu idk perhaps do threadId struct for cpu debug some time in the future...
    static inline uchar& boundary_next(int v) { return boundary_next_data[threadId * _MAX_P_ + v]; }
    static inline double4& clip(int v) { return clip_data[threadId * _MAX_P_ + v]; }
    inline  uchar& ith_plane(uchar t, int i) {return reinterpret_cast<uchar *>(&(tr(t)))[i];}
    #endif

    void compute_mesh(POINT_TYPE* pts_data, ICData& icData, InputHandler& input, OutputHandler& output) {
        std::cout << "Computing Voronoi mesh..." << std::endl;

        // define knn problem
        knn_problem *knn = NULL;

        // prepare knn problem
        int n_pts = icData.seedpos_dims[0];
        knn = knn::init((POINT_TYPE*) pts_data, n_pts);

        // solve knn problem
        knn::solve(knn);

        #ifdef VERIFY
        if (!knn::verify(knn)) {exit(EXIT_FAILURE);}
        #endif

        // write KNN data to HDF5 file
        #if defined(USE_HDF5) && defined(WRITE_KNN_OUTPUT)
        POINT_TYPE* knn_pts = knn::get_points(knn);
        unsigned int* knn_nearest = knn::get_knearest(knn);
        unsigned int* knn_permutation = knn::get_permutation(knn);

        std::string knn_output_file = input.getParameter("output_knn_file");
        if (!output.writeKNNFile(knn_output_file, knn_pts, knn_nearest, knn_permutation, icData.seedpos_dims[0], _K_)) {exit(EXIT_FAILURE);}

        free(knn_pts);
        free(knn_nearest);
        free(knn_permutation);
        #endif

        // compute voronoi cells from knn results
        std::vector<Status> stat(n_pts, security_radius_not_reached);

        // prepare mesh output data
        #ifdef USE_HDF5
        MeshCellData meshData;
        meshData.header.dimension = DIMENSION;
        meshData.header.extent = _boxsize_;
        meshData.header.n = n_pts;
        meshData.header.k = _K_;
        meshData.header.nmax = _MAX_P_;
        meshData.header.seed = 0;
        meshData.header.store_edge_coords = true;
        meshData.seeds_dims = {(hsize_t)n_pts, DIMENSION};

        compute_cells(icData.seedpos_dims[0], knn, stat, meshData);

        // set final dims for face data
        hsize_t numFaces = meshData.faces.area.size();
        meshData.faces.normal_dims = {numFaces, DIMENSION};
        hsize_t totalEdgeVerts = meshData.faces.edge_coords.size() / DIMENSION;
        meshData.faces.edge_coords_dims = {totalEdgeVerts, DIMENSION};

        // write mesh to HDF5
        std::string mesh_output_file = input.getParameter("output_mesh_file");
        if (!output.writeMeshFile(mesh_output_file, meshData)) {exit(EXIT_FAILURE);}
        #endif

        // free KNN resources
        knn::knn_free(&knn);
    }

    void compute_cells(int N_seedpts, knn_problem* knn, std::vector<Status>& stat, MeshCellData& meshData) {

        GPUBuffer<Status> gpu_stat(stat);

        int threadsPerBlock = _VORO_BLOCK_SIZE_;
        int blocksPerGrid = N_seedpts/threadsPerBlock + 1;

        cpu_compute_cell(blocksPerGrid, threadsPerBlock, N_seedpts, (double*)knn->d_stored_points, knn->d_knearests, gpu_stat.gpu_data, meshData); // add stats and output later ... :D
        
        gpu_stat.gpu2cpu();
        //for (int i = 0; i < N_seedpts; i++) {
        //    std::cout << gpu_stat.cpu_data[i] << std::endl;
        //}
    }

#ifdef CPU_DEBUG
    void cpu_compute_cell(int blocksPerGrid, int threadsPerBlock, int N_seedpts, double* d_stored_points, unsigned int* d_knearests, Status* gpu_stat, MeshCellData& meshData) {

        for (blockId = 0; blockId < blocksPerGrid; blockId++) {
            for (threadId = 0; threadId < threadsPerBlock; threadId++) {
                int seed_id = threadsPerBlock * blockId + threadId;
                if (seed_id >= N_seedpts) {break;}
                
                // compute a single voronoi cell in here...

                //create and initalize convex cell
                ConvexCell cell(seed_id, d_stored_points, &(gpu_stat[seed_id]));

                for (int v = 0; v < _K_; v++) {

                    unsigned int z = d_knearests[_K_ * seed_id + v];
                    cell.clip_by_plane(z);

                    // security radius break
                    if (cell.is_security_radius_reached(point_from_ptr3(d_stored_points + 3*z))) {break;}

                    // gpu stat return..
                    if (gpu_stat[seed_id] != success) {break;}

                }
                // check security radius again i guess
                if (!cell.is_security_radius_reached(point_from_ptr3(d_stored_points + 3 * d_knearests[_K_ * (seed_id+1) -1]))) {
                    gpu_stat[seed_id] = security_radius_not_reached;
                }

                // extract mesh data from this cell (only for successful cells)
                #ifdef USE_HDF5
                if (gpu_stat[seed_id] == success) {
                    extract_cell_mesh_data(cell, meshData);
                }
                #endif
            }
        }
    }
#endif

    ConvexCell::ConvexCell(int p_seed, double* p_pts, Status* p_status) {
        
        // define bounding box
        double eps  = .0000000001;
        double xmin = -eps;
        double ymin = -eps;
        double zmin = -eps;
        double xmax = _boxsize_ + eps;
        double ymax = _boxsize_ + eps;
        double zmax = _boxsize_ + eps;
    
        // store pointer to pts
        pts = p_pts;
    
        // set boundaries to END_OF_LIST
        first_boundary = END_OF_LIST;
        for (int i = 0; i < _MAX_P_; i++) {boundary_next(i) = END_OF_LIST;}

        // initialize plane_vid: boundary planes (-1), rest unset
        for (int i = 0; i < _MAX_P_; i++) {plane_vid[i] = -1;}

        // store seed point info
        voro_id = p_seed;
        voro_seed = make_double4(pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2], 1);

        // status set to success for now
        status = p_status;
        *status = success;

        // create 6 bounding planes for the initial bounding box
        // X-direction planes:
        clip(0) = make_double4( 1.0,  0.0,  0.0, xmin);  // x >= xmin (left face)
        clip(1) = make_double4(-1.0,  0.0,  0.0,  xmax);  // x <= xmax (right face)
    
        // Y-direction planes:
        clip(2) = make_double4( 0.0,  1.0,  0.0, ymin);  // y >= ymin (front face)
        clip(3) = make_double4( 0.0, -1.0,  0.0,  ymax);  // y <= ymax (back face)
    
        // Z-direction planes:
        clip(4) = make_double4( 0.0,  0.0,  1.0, zmin);  // z >= zmin (bottom face)
        clip(5) = make_double4( 0.0,  0.0, -1.0,  zmax);  // z <= zmax (top face)
    
        nb_v = 6;  // We now have 6 planes/vertices

        tr(0) = make_uchar3(2, 5, 0);  // Triangle 0: planes {2,5,0} define a cube corner
        tr(1) = make_uchar3(5, 3, 0);  // Triangle 1: planes {5,3,0}
        tr(2) = make_uchar3(1, 5, 2);  // Triangle 2: planes {1,5,2}
        tr(3) = make_uchar3(5, 1, 3);  // Triangle 3: planes {5,1,3}
        tr(4) = make_uchar3(4, 2, 0);  // Triangle 4: planes {4,2,0}
        tr(5) = make_uchar3(4, 0, 3);  // Triangle 5: planes {4,0,3}
        tr(6) = make_uchar3(2, 4, 1);  // Triangle 6: planes {2,4,1}
        tr(7) = make_uchar3(4, 3, 1);  // Triangle 7: planes {4,3,1}
    
        nb_t = 8;  // We now have 8 triangles
    }

    void ConvexCell::clip_by_plane(int vid) {
        
        int cur_v = new_point(vid); // add new plane equation

        if (*status == vertex_overflow) {return;}

        double4 eqn = clip(cur_v);
        nb_r = 0;

        int i = 0;
        while (i < nb_t) { // for all verticies of the cell
            if(triangle_is_in_conflict(tr(i), eqn)) {
                nb_t--;
                swap(tr(i), tr(nb_t));
                nb_r++;
            } else {
            i++;
            }
        
        }

        if (*status == needs_exact_predicates) {return;}

        if (nb_r == 0) { // if no clips, then remove the plane equation
            nb_v--;
            return;
        }

        // compute cavity boundary
        compute_boundary();
        if (*status != success) {return;}
        if (first_boundary == END_OF_LIST) {return;}

        // triangulate cavity
        uchar cir = first_boundary;
        do {
            new_triangle(cur_v, cir, boundary_next(cir));
            if (*status != success) return;
            cir = boundary_next(cir);
        } while (cir != first_boundary);
    }

    int ConvexCell::new_point(int vid) {
        if (nb_v >= _MAX_P_) { 
            *status = vertex_overflow; 
            return -1; 
        }

        double4 B = point_from_ptr3(pts + 3 * vid);
        double4 dir = minus4(voro_seed, B);
        double4 ave2 = plus4(voro_seed, B);
        double dot = dot3(ave2,dir); // TODO safe to put dot4 here, dir.w = 0
        clip(nb_v) = make_double4(dir.x, dir.y, dir.z, -dot / 2.0);
        plane_vid[nb_v] = vid;
        nb_v++;
        return nb_v - 1;
    }


    bool ConvexCell::triangle_is_in_conflict(uchar3 t, double4 eqn) const {
    double4 pi1 = clip(t.x);
    double4 pi2 = clip(t.y);
    double4 pi3 = clip(t.z);        
    
    double det = det4x4(
	pi1.x, pi2.x, pi3.x, eqn.x,
	pi1.y, pi2.y, pi3.y, eqn.y,
	pi1.z, pi2.z, pi3.z, eqn.z,
	pi1.w, pi2.w, pi3.w, eqn.w
    );    

    double maxx = std::max({std::fabs(pi1.x), std::fabs(pi2.x), std::fabs(pi3.x), std::fabs(eqn.x)});
    double maxy = std::max({std::fabs(pi1.y), std::fabs(pi2.y), std::fabs(pi3.y), std::fabs(eqn.y)});    
    double maxz = std::max({std::fabs(pi1.z), std::fabs(pi2.z), std::fabs(pi3.z), std::fabs(eqn.z)});    

    // The constant is computed by the program 
    // in predicate_generator/
    double eps = 1e-14 * maxx * maxy * maxz;
    
    double min_max;
    double max_max;
    get_minmax3(min_max, max_max, maxx, maxy, maxz);

    eps *= (max_max * max_max);

    if(std::fabs(det) < eps) {
	*status = needs_exact_predicates;
    }

    return (det > 0.0);
    }


    void ConvexCell::compute_boundary() {

        // clean circular list of the boundary
        for (int i = 0; i < _MAX_P_; i++) {
            boundary_next(i) = END_OF_LIST;
        } 
        first_boundary = END_OF_LIST;
    
        int nb_iter =0;
        uchar t = nb_t;

        while (nb_r > 0) {
            if (nb_iter++>100) {
                *status = inconsistent_boundary;
                return;
            }

            bool is_in_border[3];
            bool next_is_opp[3];

            for (int e = 0; e < 3; e++) {
                is_in_border[e] = (boundary_next(ith_plane(t, e)) != END_OF_LIST);
            }
            for (int e = 0; e < 3; e++) {
                next_is_opp[e] = (boundary_next(ith_plane(t, (e + 1)%3)) == ith_plane(t, e));
            }

            bool new_border_is_simple = true;

            // check for non manifoldness
            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) {
                    new_border_is_simple = false;   
                }
            }

            // check for more than one boundary ... or first triangle
            if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
                if (first_boundary == END_OF_LIST) {
                    for (int e = 0; e < 3; e++) {
                        boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
                    }
                    first_boundary = tr(t).x;   
                } else {
                    new_border_is_simple = false;
                }
            }

            if (!new_border_is_simple) {
                t++;
                if (t == nb_t + nb_r) {t = nb_t;}
                continue;
            }

            // link next
            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e]) {
                    boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
                }
            }

            // destroy link from removed vertices
            for (int e = 0; e < 3; e++) {
                if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
                    if (first_boundary == ith_plane(t, (e + 1) % 3)) {
                        first_boundary = boundary_next(ith_plane(t, (e + 1) % 3));
                    }
                    boundary_next(ith_plane(t, (e + 1) % 3)) = END_OF_LIST;
                }
            }
            
            //remove triangle from R, and restart iterating on R
            swap(tr(t), tr(nb_t+nb_r-1));
            t = nb_t;
            nb_r--;
        }
    }


    void ConvexCell::new_triangle(uchar i, uchar j, uchar k) {
        if (nb_t+1 >= _MAX_T_) { 
            *status = triangle_overflow; 
            return; 
        }
        tr(nb_t) = make_uchar3(i, j, k);
        nb_t++;
    }

    bool ConvexCell::is_security_radius_reached(double4 last_neig) {
        // finds furthest voro vertex distance2
        double v_dist = 0;
    
        for (int i = 0; i < nb_t; i++) {
            double4 pc = compute_triangle_point(tr(i));
            double4 diff = minus4(pc, voro_seed);
            double d2 = dot3(diff, diff);
            v_dist = std::max(d2, v_dist);
        }
    
        //compare to new neighbors distance2
        double4 diff = minus4(last_neig, voro_seed); // TODO it really should take index of the neighbor instead of the float4, then would be safe to put dot4
        double d2 = dot3(diff, diff);
        return (d2 > 4*v_dist);
    }


    double4 ConvexCell::compute_triangle_point(uchar3 t, bool persp_divide) const {
        double4 pi1 = clip(t.x);
        double4 pi2 = clip(t.y);
        double4 pi3 = clip(t.z);
        double4 result;
        result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w =  det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
        if (persp_divide) {
            return make_double4(result.x / result.w, result.y / result.w, result.z / result.w, 1);
        }
        return result;
    }



// some very basic helpers
double4 point_from_ptr3(double* f) {
    return make_double4(f[0], f[1], f[2], 1);
}

#ifdef USE_HDF5
    void extract_cell_mesh_data(ConvexCell& cell, MeshCellData& meshData) {
        // store cell id and seed position
        meshData.cell_ids.push_back(cell.voro_id);
        meshData.seeds.push_back(cell.voro_seed.x);
        meshData.seeds.push_back(cell.voro_seed.y);
        meshData.seeds.push_back(cell.voro_seed.z);

        // compute all vertex positions (each triangle = 1 vertex of the polyhedron)
        std::vector<double4> vertices(cell.nb_t);
        for (int i = 0; i < cell.nb_t; i++) {
            vertices[i] = cell.compute_triangle_point(tr(i), true);
        }

        // for each plane that appears in at least one triangle, build a face
        // a face is the polygon of vertices that share a given plane
        double cell_volume = 0.0;
        int face_count = 0;

        for (int p = 0; p < cell.nb_v; p++) {
            // collect triangle indices that reference plane p
            std::vector<int> face_tri_indices;
            for (int i = 0; i < cell.nb_t; i++) {
                if (tr(i).x == p || tr(i).y == p || tr(i).z == p) {
                    face_tri_indices.push_back(i);
                }
            }
            if (face_tri_indices.size() < 3) continue; // not a valid face

            // order vertices around the face polygon
            // two vertices are adjacent on this face if their triangles share
            // plane p and one other plane (i.e., share exactly 2 planes)
            std::vector<int> ordered;
            ordered.push_back(face_tri_indices[0]);
            std::vector<bool> used(face_tri_indices.size(), false);
            used[0] = true;

            for (size_t step = 1; step < face_tri_indices.size(); step++) {
                int last = ordered.back();
                uchar3 t_last = tr(last);

                // find the other two planes for the last triangle (besides p)
                uchar others_last[2];
                int cnt = 0;
                if (t_last.x != p) others_last[cnt++] = t_last.x;
                if (t_last.y != p) others_last[cnt++] = t_last.y;
                if (t_last.z != p) others_last[cnt++] = t_last.z;

                bool found = false;
                for (size_t j = 0; j < face_tri_indices.size(); j++) {
                    if (used[j]) continue;
                    int candidate = face_tri_indices[j];
                    uchar3 t_cand = tr(candidate);

                    // check if candidate shares exactly 2 planes with last (plane p + one other)
                    for (int o = 0; o < 2; o++) {
                        if (t_cand.x == others_last[o] || t_cand.y == others_last[o] || t_cand.z == others_last[o]) {
                            ordered.push_back(candidate);
                            used[j] = true;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                if (!found) break; // degenerate case
            }

            if (ordered.size() < 3) continue;

            // get ordered vertex positions for this face
            std::vector<double4> face_verts;
            for (int idx : ordered) {
                face_verts.push_back(vertices[idx]);
            }

            // compute face normal from clip plane equation (a, b, c)
            double4 plane_eq = clip(p);
            double normal_len = std::sqrt(plane_eq.x * plane_eq.x + plane_eq.y * plane_eq.y + plane_eq.z * plane_eq.z);
            double nx = plane_eq.x / normal_len;
            double ny = plane_eq.y / normal_len;
            double nz = plane_eq.z / normal_len;

            // ensure face vertices are oriented consistently:
            // the winding should produce a cross product aligned with the
            // outward-pointing direction (from seed toward face centroid)
            {
                double4 edge1 = minus4(face_verts[1], face_verts[0]);
                double4 edge2 = minus4(face_verts[2], face_verts[0]);
                double4 face_cross = cross3(edge1, edge2);
                // outward direction: face centroid minus seed
                double4 centroid = make_double4(0, 0, 0, 0);
                for (const auto& fv : face_verts) {
                    centroid.x += fv.x;
                    centroid.y += fv.y;
                    centroid.z += fv.z;
                }
                centroid.x /= face_verts.size();
                centroid.y /= face_verts.size();
                centroid.z /= face_verts.size();
                double4 outward = minus4(centroid, cell.voro_seed);
                if (dot3(face_cross, outward) < 0) {
                    std::reverse(face_verts.begin(), face_verts.end());
                }
            }

            // compute face area using fan triangulation from vertex 0
            double area = 0.0;
            double4 v0 = face_verts[0];
            for (size_t i = 1; i + 1 < face_verts.size(); i++) {
                double4 edge1 = minus4(face_verts[i], v0);
                double4 edge2 = minus4(face_verts[i + 1], v0);
                double4 cr = cross3(edge1, edge2);
                area += 0.5 * std::sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
            }

            // contribute to cell volume using divergence theorem:
            // V += (1/3) * sum over face triangles of (v0 - seed) . ((v1 - seed) x (v2 - seed))
            for (size_t i = 1; i + 1 < face_verts.size(); i++) {
                double4 a = minus4(face_verts[0], cell.voro_seed);
                double4 b = minus4(face_verts[i], cell.voro_seed);
                double4 c = minus4(face_verts[i + 1], cell.voro_seed);
                double4 bxc = cross3(b, c);
                cell_volume += dot3(a, bxc) / 6.0;
            }

            // neighbor cell: plane_vid[p] is the global point id for bisector planes,
            // -1 for bounding box planes
            meshData.faces.neighbor_cell.push_back(cell.plane_vid[p]);

            face_count++;

            // store normal
            meshData.faces.normal.push_back(nx);
            meshData.faces.normal.push_back(ny);
            meshData.faces.normal.push_back(nz);

            // store area
            meshData.faces.area.push_back(area);

            // store edge coordinates (ordered vertices of the face polygon)
            meshData.faces.edge_coords_offsets.push_back((int)face_verts.size());
            for (const auto& v : face_verts) {
                meshData.faces.edge_coords.push_back(v.x);
                meshData.faces.edge_coords.push_back(v.y);
                meshData.faces.edge_coords.push_back(v.z);
            }
        }

        meshData.volumes.push_back(std::fabs(cell_volume));

        // store per-cell face count
        meshData.face_counts.push_back(face_count);
    }
#endif
double4 minus4(double4 A, double4 B) {
    return make_double4(A.x-B.x, A.y-B.y, A.z-B.z, A.w-B.w);
}
double4 plus4(double4 A, double4 B) {
    return make_double4(A.x+B.x, A.y+B.y, A.z+B.z, A.w+B.w);
}
double dot4(double4 A, double4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z + A.w*B.w;
}
double dot3(double4 A, double4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z;
}
double4 mul3(double s, double4 A) {
    return make_double4(s*A.x, s*A.y, s*A.z, 1.);
}
double4 cross3(double4 A, double4 B) {
    return make_double4(A.y*B.z - A.z*B.y, A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x, 0);
}

} // namespace voronoi