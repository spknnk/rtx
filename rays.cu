#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "mpi.h"
#include "omp.h"

#define NBLOCKS dim3(128, 128)
#define NTHREADS dim3(16, 16)

#define SSSAx 2
#define PHONG_INTENS 5
#define PHONG_KD 0.8
#define PHONG_KS 0.8

#define FATAL(description)                                      \
    do {                                                        \
        std::cout << "Error in " << __FILE__ << ":" << __LINE__ \
                  << ". Message: " << description << std::endl; \
        MPI_Finalize();                                         \
        exit(0);                                                \
    } while (0)

#define CSC(call)                  \
    do {                                    \
        cudaError_t res = call;             \
        if (res != cudaSuccess) {           \
            FATAL(cudaGetErrorString(res)); \
        }                                   \
    } while (0)

#define CHECK_MPI(call)                        \
    do {                                       \
        int res = call;                        \
        if (res != MPI_SUCCESS) {              \
            char desc[MPI_MAX_ERROR_STRING];   \
            int len;                           \
            MPI_Error_string(res, desc, &len); \
            FATAL(desc);                       \
        }                                      \
    } while (0)

#define OpenMP 0
#define CUDA 1
int parallelizationMode;

void parse_flags(int argc, char *argv[]) {
   
    if (argc < 2) return;
    if (argc == 2) {
        if (strcmp(argv[1], "--gpu") == 0) {
            parallelizationMode = CUDA;
        }
        
        if (strcmp(argv[1], "--cpu") == 0) {
            parallelizationMode = OpenMP;
        }
        
        else if ((strcmp(argv[1], "--cpu") != 0) and (strcmp(argv[1], "--gpu") != 0)) {
            std::cout << "Unknown args" << std::endl;
            MPI_Finalize();
            exit(0);
        }
    }
    if (argc > 2) {
        std::cout << "A lot of args" << std::endl;
        MPI_Finalize();
        exit(0);
    }
}

struct MPIContext {
    MPIContext(int *argc, char ***argv) { CHECK_MPI(MPI_Init(argc, argv)); }
    ~MPIContext() {
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
        CHECK_MPI(MPI_Finalize());
    }
};

template <typename T>
struct Vector3 {
    T x, y, z;

    __host__ __device__ Vector3(T x = T{}, T y = T{}, T z = T{})
        : x(x), y(y), z(z) {}

    friend std::istream &operator>>(std::istream &is, Vector3 &v) {
        is >> v.x >> v.y >> v.z;
        return is;
    }
};

struct CameraMovement {
    double r0, z0, phi0, ar, az, wr, wz, wphi, pr, pz;

    friend std::istream &operator>>(std::istream &is, CameraMovement &p) {
        is >> p.r0 >> p.z0 >> p.phi0 >> p.ar >> p.az >> p.wr >> p.wz >> p.wphi >> p.pr >> p.pz;
        return is;
    }
};

struct FigureParams {
    Vector3<double> center, color;
    double radius, k_refl, k_refr;
    int lights_num;

    friend std::istream &operator>>(std::istream &is, FigureParams &p) {
        is >> p.center >> p.color >> p.radius >> p.k_refl >> p.k_refr >> p.lights_num;
        return is;
    }
};

struct FloorParams {
    Vector3<double> a, b, c, d, color;
    double k_refl;
    std::string texture_path;

    friend std::istream &operator>>(std::istream &is, FloorParams &p) {
        is >> p.a >> p.b >> p.c >> p.d >> p.texture_path >> p.color >> p.k_refl;
        return is;
    }
};

struct LightParams {
    Vector3<double> pos;
    Vector3<double> color;

    friend std::istream &operator>>(std::istream &is, LightParams &p) {
        is >> p.pos >> p.color;
        return is;
    }
};

struct Params {
    int nframes, w, h, lights_num;
    double angle;
    CameraMovement camera_center, camera_dir;
    FigureParams hex, octa, icos;
    FloorParams floor;
    
    std::string output_pattern;
    std::vector<LightParams> lights;

    friend std::istream &operator>>(std::istream &is, Params &p) {
        is >> p.nframes >> p.output_pattern >> p.w >> p.h >> p.angle >> p.camera_center >> p.camera_dir >> p.hex >> p.octa >> p.icos >> p.floor >> p.lights_num;
        p.lights.resize(p.lights_num);
        for (auto &it : p.lights) is >> it;
        return is;
    }
};

struct Triangle {
    Vector3<double> a, b, c, color;
};

template <typename T>
__host__ __device__ T min(const T &a, const T &b) {
    if (a < b) return a;
    return b;
}

template <typename T>
__host__ __device__ T max(const T &a, const T &b) {
    if (a > b) return a;
    return b;
}

__host__ __device__ double dot_product(const Vector3<double> &a, const Vector3<double> &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vector3<double> cross_product(const Vector3<double> &a, const Vector3<double> &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,a.x * b.y - a.y * b.x};
}

__host__ __device__ double norm(const Vector3<double> &v) {
    return sqrt(dot_product(v, v));
}

__host__ __device__ Vector3<double> normalize(const Vector3<double> &v) {
    double l = norm(v);
    return {v.x / l, v.y / l, v.z / l};
}

__host__ __device__ Vector3<double> diff(const Vector3<double> &a, const Vector3<double> &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ Vector3<double> add(const Vector3<double> &a, const Vector3<double> &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ Vector3<double> mult(const Vector3<double> &a, const Vector3<double> &b) {
    return {b.x * a.x, b.y * a.y, b.z * a.z};
}

__host__ __device__ Vector3<double> mult(const Vector3<double> &a, double k) {
    return {k * a.x, k * a.y, k * a.z};
}

__host__ __device__ Vector3<double> mult(const Vector3<double> &a, const Vector3<double> &b, const Vector3<double> &c, const Vector3<double> &d) {
    return {a.x * d.x + b.x * d.y + c.x * d.z,
            a.y * d.x + b.y * d.y + c.y * d.z,
            a.z * d.x + b.z * d.y + c.z * d.z};
}

__host__ __device__ Vector3<double> inverse(const Vector3<double> &v) {
    return {-v.x, -v.y, -v.z};
}

__host__ __device__ Vector3<double> div(const Vector3<double> &a, double k) {
    return {a.x / k, a.y / k, a.z / k};
}

__host__ __device__ Vector3<double> reflect(const Vector3<double> &v, const Vector3<double> &n) {
    return diff(v, mult(n, 2.0 * dot_product(v, n)));
}

__host__ __device__ uchar4 color_from_normalized(const Vector3<double> &v) {
    double x = min(v.x, 1.);
    x = max(x, 0.);
    double y = min(v.y, 1.);
    y = max(y, 0.);
    double z = min(v.z, 1.);
    z = max(z, 0.);
    return make_uchar4(255. * x, 255. * y, 255. * z, 0u);
}

std::vector<std::string> split_string(const std::string &s, char d) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string word;
    while (getline(ss, word, d)) {
        result.push_back(word);
    }
    return result;
}

void importObj(std::vector<Triangle> &scene_Triangles, const std::string &filepath, const FigureParams &fp) {
    std::ifstream is(filepath);
    if (!is) {
        std::string desc = "can't open " + filepath;
        FATAL(desc);
    }

    /*
    std::vector<Triangle> triangle_list(10);
    triangle_list[0] = Triangle{Vector3<double> {}, Vector3<double> {}, Vector3<double> {}, fp.color}
    */

    double r = 0;
    std::vector<Vector3<double>> vertices;
    std::vector<Triangle> figure_Triangles;
    std::string line;
    while (std::getline(is, line)) {
        std::vector<std::string> buffer = split_string(line, ' ');
        if (line.empty()) {
            continue;
        } else if (buffer[0] == "v") {
            double x = std::stod(buffer[2]);
            double y = std::stod(buffer[3]);
            double z = std::stod(buffer[4]);

            vertices.push_back({x, y, z});
        } else if (buffer[0] == "f") {
            std::vector<std::string> indexes = split_string(buffer[1], '/');
            Vector3<double> a = vertices[std::stoi(indexes[0]) - 1];
            indexes = split_string(buffer[2], '/');
            Vector3<double> b = vertices[std::stoi(indexes[0]) - 1];
            indexes = split_string(buffer[3], '/');
            Vector3<double> c = vertices[std::stoi(indexes[0]) - 1];

            r = max(r, norm(a));
            r = max(r, norm(b));
            r = max(r, norm(c));

            figure_Triangles.push_back(Triangle{a, b, c, fp.color});
        }
    }

    double radius = fp.radius / r;
    for (auto &single_triangle : figure_Triangles) {
        scene_Triangles.push_back({add(mult(single_triangle.a, radius), fp.center), add(mult(single_triangle.b, radius), fp.center), add(mult(single_triangle.c, radius), fp.center), single_triangle.color});
    }
}



void add_floor_to_scene(std::vector<Triangle> &scene_Triangles, const FloorParams &fp) {
    scene_Triangles.push_back({fp.c, fp.b, fp.a, fp.color});
    scene_Triangles.push_back({fp.a, fp.d, fp.c, fp.color});
}

struct Mat3d {
    double m[3][3];
    __host__ __device__ Mat3d(double m11 = 0, double m12 = 0, double m13 = 0,
                              double m21 = 0, double m22 = 0, double m23 = 0,
                              double m31 = 0, double m32 = 0, double m33 = 0) {
        m[0][0] = m11;
        m[0][1] = m12;
        m[0][2] = m13;
        m[1][0] = m21;
        m[1][1] = m22;
        m[1][2] = m23;
        m[2][0] = m31;
        m[2][1] = m32;
        m[2][2] = m33;
    }
};

__host__ __device__ double det(const Mat3d &m) {
    return m.m[0][0] * m.m[1][1] * m.m[2][2] +
           m.m[1][0] * m.m[0][2] * m.m[2][1] +
           m.m[2][0] * m.m[0][1] * m.m[1][2] -
           m.m[0][2] * m.m[1][1] * m.m[2][0] -
           m.m[0][0] * m.m[1][2] * m.m[2][1] -
           m.m[0][1] * m.m[1][0] * m.m[2][2];
}

__host__ __device__ Mat3d inverse(const Mat3d &m) {
    double d = det(m);

    double m11 = (m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2]) / d;
    double m12 = (m.m[2][1] * m.m[0][2] - m.m[0][1] * m.m[2][2]) / d;
    double m13 = (m.m[0][1] * m.m[1][2] - m.m[1][1] * m.m[0][2]) / d;

    double m21 = (m.m[2][0] * m.m[1][2] - m.m[1][0] * m.m[2][2]) / d;
    double m22 = (m.m[0][0] * m.m[2][2] - m.m[2][0] * m.m[0][2]) / d;
    double m23 = (m.m[1][0] * m.m[0][2] - m.m[0][0] * m.m[1][2]) / d;

    double m31 = (m.m[1][0] * m.m[2][1] - m.m[2][0] * m.m[1][1]) / d;
    double m32 = (m.m[2][0] * m.m[0][1] - m.m[0][0] * m.m[2][1]) / d;
    double m33 = (m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1]) / d;

    return Mat3d(m11, m12, m13, m21, m22, m23, m31, m32, m33);
}

__host__ __device__ Vector3<double> mult(const Mat3d &m, const Vector3<double> &v) {
    Vector3<double> res;
    res.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z;
    res.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z;
    res.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z;
    return res;
}

__host__ __device__ void triangle_intersection(const Vector3<double> &origin, const Vector3<double> &dir, const Triangle &Triangle, double *t, double *u, double *v) {
    Vector3<double> e1 = diff(Triangle.b, Triangle.a);
    Vector3<double> e2 = diff(Triangle.c, Triangle.a);

    Mat3d m(-dir.x, e1.x, e2.x, -dir.y, e1.y, e2.y, -dir.z, e1.z, e2.z);
    Vector3<double> tmp = mult(inverse(m), diff(origin, Triangle.a));

    *t = tmp.x;
    *u = tmp.y;
    *v = tmp.z;
}

__host__ __device__ bool shadow_ray_hit(const Vector3<double> &origin, const Vector3<double> &dir, const Triangle *scene_Triangles, int nTriangles, double *hit_t) {
    double t_min = 1 / 0.;
    bool hit = false;
    for (int i = 0; i < nTriangles; ++i) {
        auto Triangle = scene_Triangles[i];
        double t, u, v;
        triangle_intersection(origin, dir, Triangle, &t, &u, &v);
        if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0) {
            if (t < t_min) {
                t_min = t;
            }
            hit = true;
        }
    }
    *hit_t = t_min;
    return hit;
}

__host__ __device__ Vector3<double> phong_model(const Vector3<double> &pos, const Vector3<double> &dir, const Triangle &TriangleObj, const Triangle *scene_Triangles, int nTriangles, const LightParams *lights, int lights_num) {
    Vector3<double> normal = normalize(cross_product(diff(TriangleObj.b, TriangleObj.a), diff(TriangleObj.c, TriangleObj.a)));

    Vector3<double> ambient{0.1, 0.1, 0.1};
    Vector3<double> diffuse{0., 0., 0.};
    Vector3<double> specular{0., 0., 0.};

    for (int i = 0; i < lights_num; ++i) {
        Vector3<double> light_pos = lights[i].pos;
        Vector3<double> L = diff(light_pos, pos);
        double d = norm(L);
        L = normalize(L);

        double hit_t = 0.0;
        if (shadow_ray_hit(light_pos, inverse(L), scene_Triangles, nTriangles, &hit_t) && (hit_t > d || (hit_t > d || (d - hit_t < 0.0005)))) {
            double k = PHONG_INTENS / (d + 0.001f);
            diffuse = add(diffuse, mult(lights[i].color, max(PHONG_KD * k * dot_product(L, normal), 0.0)));

            Vector3<double> R = normalize(reflect(inverse(L), normal));
            Vector3<double> S = inverse(dir);
            specular = add(specular, mult(lights[i].color, PHONG_KS * k * std::pow(max(dot_product(R, S), 0.0), 32)));
        }
    }
    return add(add(mult(ambient, TriangleObj.color), mult(diffuse, TriangleObj.color)), mult(specular, TriangleObj.color));
}

__host__ __device__ uchar4 ray(const Vector3<double> &pos, const Vector3<double> &dir, const Triangle *scene_Triangles, int nTriangles, LightParams *lights, int lights_num) {
    int k, k_min = -1;
    double ts_min;
    for (k = 0; k < nTriangles; k++) {
        Vector3<double> e1 = diff(scene_Triangles[k].b, scene_Triangles[k].a);
        Vector3<double> e2 = diff(scene_Triangles[k].c, scene_Triangles[k].a);
        Vector3<double> p = cross_product(dir, e2);
        double div = dot_product(p, e1);
        if (fabs(div) < 1e-10) continue;
        Vector3<double> t = diff(pos, scene_Triangles[k].a);
        double u = dot_product(p, t) / div;
        if (u < 0.0 || u > 1.0) continue;
        Vector3<double> q = cross_product(t, e1);
        double v = dot_product(q, dir) / div;
        if (v < 0.0 || v + u > 1.0) continue;
        double ts = dot_product(q, e2) / div;
        if (ts < 0.0) continue;
        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }
    if (k_min == -1) return {0, 0, 0, 0};
    return color_from_normalized(phong_model(add(mult(dir, ts_min), pos), dir, scene_Triangles[k_min], scene_Triangles, nTriangles, lights, lights_num));
}

void getRenderOMP(uchar4 *data, int w, int h, Vector3<double> pc, Vector3<double> pv, double angle, const Triangle *scene_Triangles, int nTriangles, LightParams *lights, int lights_num) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    Vector3<double> bz = normalize(diff(pv, pc));
    Vector3<double> bx = normalize(cross_product(bz, {0.0, 0.0, 1.0}));
    Vector3<double> by = normalize(cross_product(bx, bz));
#pragma omp parallel for
    for (int pix = 0; pix < w * h; ++pix) {
        int i = pix % w;
        int j = pix / w;
        Vector3<double> v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
        Vector3<double> dir = mult(bx, by, bz, v);
        data[(h - 1 - j) * w + i] = ray(pc, normalize(dir), scene_Triangles, nTriangles, lights, lights_num);
    }
}

__global__ void getRenderCUDA(uchar4 *data, int w, int h, Vector3<double> pc, Vector3<double> pv, double angle, const Triangle *scene_Triangles, int nTriangles, LightParams *lights, int lights_num) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    Vector3<double> bz = normalize(diff(pv, pc));
    Vector3<double> bx = normalize(cross_product(bz, {0.0, 0.0, 1.0}));
    Vector3<double> by = normalize(cross_product(bx, bz));
    for (int j = id_y; j < h; j += offset_y)
        for (int i = id_x; i < w; i += offset_x) {
            Vector3<double> v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
            Vector3<double> dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] =
                ray(pc, normalize(dir), scene_Triangles, nTriangles, lights, lights_num);
        }
}

__host__ __device__ uchar4 SSAA(uchar4 *data, int i, int j, int w, int h, int kernel_w, int kernel_h) {
    Vector3<double> res;
    for (int y = i; y < i + kernel_h; ++y)
        for (int x = j; x < j + kernel_w; ++x) {
            auto pix = data[y * w + x];
            res = add(res, Vector3<double>{(double)pix.x, (double)pix.y, (double)pix.z});
        }
    auto pix = div(res, kernel_w * kernel_h);
    return make_uchar4(pix.x, pix.y, pix.z, 0);
}

__global__ void getSSAA_CUDA(uchar4 *dst, uchar4 *src, int new_w, int new_h, int w, int h) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    int kernel_w = w / new_w;
    int kernel_h = h / new_h;

    for (int i = id_y; i < new_h; i += offset_y) {
        for (int j = id_x; j < new_w; j += offset_x) {
            int pix_i = i * kernel_h;
            int pix_j = j * kernel_w;

            dst[i * new_w + j] = SSAA(src, pix_i, pix_j, w, h, kernel_w, kernel_h);
        }
    }
}

void getSSAA_OMP(uchar4 *dst, uchar4 *src, int new_w, int new_h, int w, int h) {
    int kernel_w = w / new_w;
    int kernel_h = h / new_h;
#pragma omp parallel for
    for (int pix = 0; pix < new_w * new_h; ++pix) {
        int i = pix / new_w;
        int j = pix % new_w;

        int pix_i = i * kernel_h;
        int pix_j = j * kernel_w;

        dst[i * new_w + j] = SSAA(src, pix_i, pix_j, w, h, kernel_w, kernel_h);
    }
}

void getCameraPos(const CameraMovement &c, const CameraMovement &n, double t, Vector3<double> *pc, Vector3<double> *pv) {
    double phic = c.phi0 + c.wphi * t, phin = n.phi0 + n.wphi * t;
    double rc = c.r0 + c.ar * sin(c.wr * t + c.pr), zc = c.z0 + c.ar * sin(c.wz * t + c.pz);
    double rn = n.r0 + n.ar * sin(n.wr * t + n.pr), zn = n.z0 + n.ar * sin(n.wz * t + n.pz);

    *pv = Vector3<double>{rn * cos(phin), rn * sin(phin), zn};
    *pc = Vector3<double>{rc * cos(phic), rc * sin(phic), zc};
}

void write_image(const std::string &path, const std::vector<uchar4> &data, int w, int h) {
    MPI_File file;
    CHECK_MPI(MPI_File_open(MPI_COMM_SELF, path.data(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file));
    CHECK_MPI(MPI_File_write(file, &w, 1, MPI_INT, MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_File_write(file, &h, 1, MPI_INT, MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_File_write(file, data.data(), sizeof(uchar4) * w * h,  MPI_BYTE, MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_File_close(&file));
}

void signal_handler(int signal){
    std::cout << "Error. Bad signal: " << signal << std::endl;
    MPI_Finalize();
    exit(0);
}

int main(int argc, char *argv[]) {

    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);

    MPIContext ctx(&argc, &argv);
    parse_flags(argc, argv);

    int rank, nprocesses;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nprocesses));

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    std::vector<Triangle> scene_Triangles;
    Params params;
    if (rank == 0) {
        std::cin >> params;
        importObj(scene_Triangles, "hex.obj", params.hex);
        importObj(scene_Triangles, "octa.obj", params.octa);
        importObj(scene_Triangles, "icos.obj", params.icos);
        scene_Triangles.push_back({params.floor.c, params.floor.b, params.floor.a, params.floor.color});
        scene_Triangles.push_back({params.floor.a, params.floor.d, params.floor.c, params.floor.color});
    }

    int output_pattern_size = params.output_pattern.size();
    CHECK_MPI(MPI_Bcast(&params.nframes, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&output_pattern_size, 1, MPI_INT, 0, MPI_COMM_WORLD));
    params.output_pattern.resize(output_pattern_size);
    CHECK_MPI(MPI_Bcast((char *)params.output_pattern.data(), output_pattern_size, MPI_CHAR, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.w, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.h, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.angle, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.camera_center, sizeof(CameraMovement), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.camera_dir, sizeof(CameraMovement), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.hex, sizeof(FigureParams), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.octa, sizeof(FigureParams), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.icos, sizeof(FigureParams), MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_MPI(MPI_Bcast(&params.floor.a, sizeof(Vector3<double>), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.floor.b, sizeof(Vector3<double>), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.floor.c, sizeof(Vector3<double>), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.floor.d, sizeof(Vector3<double>), MPI_BYTE, 0, MPI_COMM_WORLD));
    int texture_path_size = params.floor.texture_path.size();
    CHECK_MPI(MPI_Bcast(&texture_path_size, 1, MPI_INT, 0, MPI_COMM_WORLD));
    params.floor.texture_path.resize(texture_path_size);
    CHECK_MPI(MPI_Bcast((char *)params.floor.texture_path.data(), texture_path_size, MPI_CHAR, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.floor.color, sizeof(Vector3<double>), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&params.floor.k_refl, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    params.lights.resize(params.lights_num);
    CHECK_MPI(MPI_Bcast(&params.lights_num, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(params.lights.data(), sizeof(LightParams) * params.lights_num, MPI_BYTE, 0, MPI_COMM_WORLD));

    int nTriangles = scene_Triangles.size();
    CHECK_MPI(MPI_Bcast(&nTriangles, 1, MPI_INT, 0, MPI_COMM_WORLD));
    scene_Triangles.resize(nTriangles);
    CHECK_MPI(MPI_Bcast(scene_Triangles.data(), sizeof(Triangle) * nTriangles, MPI_BYTE, 0, MPI_COMM_WORLD));

    int ndevices;
    CSC(cudaGetDeviceCount(&ndevices));
    CSC(cudaSetDevice(rank % ndevices));

    Triangle *gpu_scene_Triangles;
    LightParams *gpu_lights;
    if (parallelizationMode == CUDA) {
        auto Triangles_size = sizeof(Triangle) * scene_Triangles.size();
        CSC(cudaMalloc(&gpu_scene_Triangles, Triangles_size));
        CSC(cudaMemcpy(gpu_scene_Triangles, scene_Triangles.data(), Triangles_size, cudaMemcpyHostToDevice));
        auto lights_size = sizeof(LightParams) * params.lights_num;
        CSC(cudaMalloc(&gpu_lights, lights_size));
        CSC(cudaMemcpy(gpu_lights, params.lights.data(), lights_size, cudaMemcpyHostToDevice));
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    std::vector<uchar4> data_render(SSSAx*params.w * SSSAx*params.h), data_ssaa(params.w * params.h);
    uchar4 *gpu_data_render, *gpu_data_ssaa;
    if (parallelizationMode == CUDA) {
        CSC(cudaMalloc(&gpu_data_render, sizeof(uchar4) * SSSAx*params.w * SSSAx*params.h));
        CSC(cudaMalloc(&gpu_data_ssaa, sizeof(uchar4) * params.w * params.h));
    }
    for (int frame = rank; frame < params.nframes; frame += nprocesses) {
        Vector3<double> pc, pv;
        getCameraPos(params.camera_center, params.camera_dir, 0.1 * (double)frame, &pc, &pv);
        auto start = std::chrono::high_resolution_clock::now();

        if (parallelizationMode == OpenMP) {
            getRenderOMP(data_render.data(), SSSAx*params.w, SSSAx*params.h, pc, pv, params.angle, scene_Triangles.data(), scene_Triangles.size(), params.lights.data(), params.lights.size());
            getSSAA_OMP(data_ssaa.data(), data_render.data(), params.w, params.h, SSSAx*params.w, SSSAx*params.h);
        }

        if (parallelizationMode == CUDA) {
            getRenderCUDA<<<NBLOCKS, NTHREADS>>>(gpu_data_render, SSSAx*params.w, SSSAx*params.h, pc, pv, params.angle,gpu_scene_Triangles, scene_Triangles.size(), gpu_lights, params.lights.size());
            CSC(cudaDeviceSynchronize());
            getSSAA_CUDA<<<NBLOCKS, NTHREADS>>>(gpu_data_ssaa, gpu_data_render, params.w, params.h, SSSAx*params.w, SSSAx*params.h);
            CSC(cudaMemcpy(data_ssaa.data(), gpu_data_ssaa,sizeof(uchar4) * params.w * params.h, cudaMemcpyDeviceToHost));
        }

        char output_path[256];
        sprintf(output_path, params.output_pattern.data(), frame);
        write_image(output_path, data_ssaa, params.w, params.h);
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << frame << "\t" << output_path << "\t" << time.count() << "ms" << std::endl;
    }

    if (parallelizationMode == CUDA) {
        CSC(cudaFree(gpu_scene_Triangles));
        CSC(cudaFree(gpu_lights));
        CSC(cudaFree(gpu_data_ssaa));
        CSC(cudaFree(gpu_data_render));
    }

    return 0;
}