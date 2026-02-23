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
        double4 voro_seed;
        uchar first_boundary;
        Status* status;
        uchar nb_v;
        uchar nb_t;
        uchar nb_r;
        int plane_vid[_MAX_P_]; // maps plane index to global point id (-1 for boundary planes)

        void clip_by_plane(int vid);
        int new_point(int vid);
        bool triangle_is_in_conflict(uchar3 t, double4 eqn) const;
        void compute_boundary();
        void new_triangle(uchar i, uchar j, uchar k);
        bool is_security_radius_reached(double4 last_neig);
        double4 compute_triangle_point(uchar3 t, bool persp_divide=true) const;
    };

    
    // prob need a different struct to store the actual mesh then

    void compute_mesh(POINT_TYPE* pts_data, ICData& icData, InputHandler& input, OutputHandler& output);

    void compute_cells(int N_seedpts, knn_problem* knn, std::vector<Status>& stat, MeshCellData& meshData);
    
    void cpu_compute_cell(int blocksPerGrid, int threadsPerBlock, int N_seedpts, double* d_stored_points, unsigned int* d_knearests, Status* gpu_stat, MeshCellData& meshData);

    void extract_cell_mesh_data(ConvexCell& cell, MeshCellData& meshData);


    // many inline helpers....
    double4 point_from_ptr3(double* f);
    double4 minus4(double4 A, double4 B);
    double4 plus4(double4 A, double4 B);
    double dot4(double4 A, double4 B);
    double dot3(double4 A, double4 B);
    double4 mul3(double s, double4 A);
    double4 cross3(double4 A, double4 B);
    
    inline double det2x2(double a11, double a12, double a21, double a22) {
        return a11*a22 - a12*a21;
    }

    inline double det3x3(double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33) {
        return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
    }

    inline double det4x4(
        double a11, double a12, double a13, double a14,
        double a21, double a22, double a23, double a24,               
        double a31, double a32, double a33, double a34,  
        double a41, double a42, double a43, double a44) {

        double m12 = a21*a12 - a11*a22;
        double m13 = a31*a12 - a11*a32;
        double m14 = a41*a12 - a11*a42;
        double m23 = a31*a22 - a21*a32;
        double m24 = a41*a22 - a21*a42;
        double m34 = a41*a32 - a31*a42;
    
        double m123 = m23*a13 - m13*a23 + m12*a33;
        double m124 = m24*a13 - m14*a23 + m12*a43;
        double m134 = m34*a13 - m14*a33 + m13*a43;
        double m234 = m34*a23 - m24*a33 + m23*a43;
    
        return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
    }

    template <typename T> void inline swap(T& a, T& b) { T c(a); a = b; b = c; }

    inline void get_minmax3(double& m, double& M, double x1, double x2, double x3) {
        m = std::min(std::min(x1, x2), x3);
        M = std::max(std::max(x1, x2), x3);
    }

} // namespace voronoi

#endif // VORONOI_H
