#ifndef VORONOI_H
#define VORONOI_H

#include <string>
#include "global/allvars.h"
#include "../knn/knn.h"
#include "../io/input.h"
#include "../io/output.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>

/* 
 * This part of the code is heavily inspired by the work of: Nicolas Ray, Dmitry Sokolov, 
 * Sylvain Lefebvre, Bruno L'evy, "Meshless Voronoi on the GPU", ACM Trans. Graph., 
 * vol. 37, no. 6, Dec. 2018. If you build upon this code, we recommend  
 * reading and citing their paper: https://doi.org/10.1145/3272127.3275092
 */

// probably add a voronoi mesh struct here (such that the output mesh data is seperate from the data the mesh stores)

// not used yet but this will be a mesh struct for computation?
struct VMesh {
    hsize_t* cell_ids; // cell ids
    double3* seeds; // seedpoints
    hsize_t n_seeds; // number of cells
    double* volumes; // area in 2D, volume in 3D
    hsize_t* face_counts; // number of faces per cell
    hsize_t* face_ptr; // pointer to start of each cell's faces in the face arrays
    int* neighbor_cell; // global id of neighboring cell for each face (how will this work with ghost cells and periodic??)
    double3* face_normal; // normal vector for each face
    double* face_area; // edge length in 2D, face area in 3D
    hsize_t num_faces; // total number of faces in the mesh
    #ifdef DEBUG_MODE
    double* edge_coords; // flat array of all face vertex coordinates (DIMENSION doubles per vertex)
    hsize_t* edge_coords_offsets; // number of vertices per face
    hsize_t num_edge_coord_verts; // total number of edge coord vertices
    #endif
};


namespace voronoi {

    enum Status {
        triangle_overflow = 0,
        vertex_overflow = 1,
        inconsistent_boundary = 2,
        security_radius_not_reached = 3,
        success = 4,
        needs_exact_predicates = 5
    };

    template <class T> struct GPUBuffer {
        void init(T* data) {
            cpu_data = data;
            gpuMalloc((void**)& gpu_data, size * sizeof(T));
            cpu2gpu();
        }
        GPUBuffer(std::vector<T>& v) {size = v.size() ;init(v.data());}
        ~GPUBuffer() { gpuFree(gpu_data); }

        void cpu2gpu() { gpuMemcpy(gpu_data, cpu_data, size * sizeof(T)); }
        void gpu2cpu() { gpuMemcpy(cpu_data, gpu_data, size * sizeof(T)); }

        T* cpu_data;
        T* gpu_data;
        int size;
    };

    // struct used for mesh generation
    struct ConvexCell {
        ConvexCell(int p_seed, double* p_pts, Status* p_status);

        double *pts;
        int voro_id;
        double4 voro_seed; // double4 so all math helpers (minus4, dot3, ...) work uniformly.
                          // In 2D: z=0, w=1 (set by point_from_ptr). The z component is
                          // harmless in dot3/cross3 since it's zero.
        uchar first_boundary;
        Status* status;
        uchar nb_v;
        uchar nb_t; // number of cell vertices (3-plane intersections in 3D, 2-line intersections in 2D)
        uchar nb_r;
        int plane_vid[_MAX_P_]; // maps plane index to global point id (-1 for boundary planes)

        void clip_by_plane(int vid);
        int new_halfplane(int vid);
        void compute_boundary();
        bool is_security_radius_reached(double4 last_neig);

        // unified 2D/3D vertex operations (VERT_TYPE = uchar2 in 2D, uchar3 in 3D)
        bool vert_is_in_conflict(VERT_TYPE v, double4 eqn) const;
        void new_vertex(uchar i, uchar j, uchar k = 0);
        double4 compute_vertex_point(VERT_TYPE v, bool persp_divide=true) const;
    };

    // allocation and deallocation of VMesh
    VMesh* allocate_vmesh(hsize_t n_seeds, hsize_t initial_face_capacity);
    void free_vmesh(VMesh* mesh);

    // main mesh computation (returns VMesh*)
    VMesh* compute_mesh(POINT_TYPE* pts_data, ICData& icData, InputHandler& input, OutputHandler& output);
    void compute_cells(int N_seedpts, knn_problem* knn, std::vector<Status>& stat, VMesh* mesh);

    #ifdef CPU_DEBUG
    void cpu_compute_cell(int blocksPerGrid, int threadsPerBlock, int N_seedpts, double* d_stored_points, unsigned int* d_knearests, Status* gpu_stat, VMesh* mesh, hsize_t& face_capacity);
    #endif

    void extract_cell_to_vmesh(ConvexCell& cell, VMesh* mesh, hsize_t cell_index, hsize_t& face_capacity);

    #ifdef USE_HDF5
    void vmesh_to_meshdata(VMesh* mesh, MeshCellData& meshData);
    #endif

    double compute_cell_area_2d(const std::vector<double4>& vertices, int nb_t);
    bool collect_face_vertices(ConvexCell& cell, int p, const std::vector<double4>& vertices, std::vector<double4>& face_verts);
    double3 compute_face_normal(int p);
    double compute_face_measure(std::vector<double4>& face_verts, double4 seed, double* cell_volume);
    void store_face_data(VMesh* mesh, const std::vector<double4>& face_verts, int neighbor_id, double3 normal, double face_measure);

    void unpermute_vmesh(VMesh* mesh, const unsigned int* sorted_to_original);


} // namespace voronoi

#endif // VORONOI_H
