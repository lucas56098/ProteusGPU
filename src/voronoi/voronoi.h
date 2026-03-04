#ifndef VORONOI_H
#define VORONOI_H

#include <string>
#include "../global/allvars.h"
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


// voronoi mesh struct used for hydro solver
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

    hsize_t n_hydro; // number of active cells (n_ghost = n_seeds - n_hydro)
    hsize_t* ghost_ids; // ids of the corresponding original cell (i.e. the ghost cell with id cell_ids[(n_hydro-1) + 4] has ghost_ids[4])
};

namespace voronoi {

    // buffer struct for GPU-CPU data transfer (maybe we dont need this on GH200?)
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

    // allocation and deallocation of VMesh
    VMesh* allocate_vmesh(hsize_t n_seeds, hsize_t initial_face_capacity);
    void free_vmesh(VMesh* mesh);

    // main mesh computation
    VMesh* compute_mesh(POINT_TYPE* pts_data, int num_points);
    void compute_cells(int N_seedpts, knn_problem* knn, std::vector<Status>& stat, VMesh* mesh);

    // kernels
    #ifdef CPU_DEBUG
    void cpu_compute_cell(int blocksPerGrid, int threadsPerBlock, int N_seedpts, double* d_stored_points, unsigned int* d_knearests, Status* gpu_stat, VMesh* mesh, hsize_t& face_capacity);
    #endif

    // restore original input pts order (after KNN sorted it...)
    void unpermute_vmesh(VMesh* mesh, const unsigned int* sorted_to_original);

    #ifdef USE_HDF5
    // convert VMesh (for hydro computation) to MeshCellData (for output)
    void vmesh_to_meshdata(VMesh* mesh, MeshCellData& meshData);
    #endif

} // namespace voronoi

#endif // VORONOI_H
