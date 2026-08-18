#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <cub/cub.cuh>
#include <cuinterval/cuinterval.h>
#include <thrust/iterator/transform_iterator.h>

// For a range of subdomains/widths, record the bounds that interval
// refinement produces over an objective's parent domain. Backend-agnostic
// with respect to *which* objective: each problem is an Objective struct
// (VARS, symmetry-fixed dims, device/host evaluators, parent-domain bounds)
// selected at runtime via a CLI arg; see the Objective structs below and
// main()'s dispatch.

// Requires:
// 1. A kernel to determine bounds over a subdomain, given a subregion index
// 2. Min and Max reduction over those bounds to determine the refinement


namespace
{

void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Writes `values` as a raw, native-byte-order float64 array -- no header, since
// the reader (study/variable_subdomain/plot_distributions.py) infers the
// element count from the file size.
void write_f64_array(const std::filesystem::path &path, const std::vector<double> &values) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "failed to open %s for writing\n", path.c_str());
        std::exit(1);
    }
    std::fwrite(values.data(), sizeof(double), values.size(), f);
    std::fclose(f);
}
}


/**
 * @brief Maps a thread id (linear subregion index) to per-dimension bounds,
 * under a uniform N-way subdivision of every free variable's bounds.
 * Dims where Objective::is_symmetry_fixed(d) holds are copied straight from
 * `parent` and consume no digit of `sidx` -- they're already point
 * intervals, so subdividing them would only inflate the subregion count for
 * identical copies.
 *
 * @tparam T
 * @tparam Objective Problem definition: VARS and is_symmetry_fixed(d)
 * @param parent Parent domain bounds, Objective::VARS entries
 * @param N Subdivision per free variable
 * @param sidx Subregion index, in [0, N^free_var_count<Objective>())
 * @param out Objective::VARS entries
 */
template<typename T, typename Objective>
__host__ __device__ inline void subregion_bounds(const cu::interval<T> *parent,
                                                   std::size_t N,
                                                   std::size_t sidx,
                                                   cu::interval<T> *out) {
    for (std::size_t d = 0; d < Objective::VARS; ++d) {
        if (Objective::is_symmetry_fixed(d)) {
            out[d] = parent[d];
            continue;
        }

        std::size_t part = sidx % N;
        sidx /= N;

        T width = (parent[d].ub - parent[d].lb) / static_cast<T>(N);
        out[d].lb = parent[d].lb + width * static_cast<T>(part);
        out[d].ub = (part + 1 == N) ? parent[d].ub
                                     : parent[d].lb + width * static_cast<T>(part + 1);
    }
}

// Free-variable count: Objective::VARS less however many dims
// Objective::is_symmetry_fixed pins to a point interval.
template<typename Objective>
__host__ __device__ constexpr std::size_t free_var_count() {
    std::size_t count = 0;
    for (std::size_t d = 0; d < Objective::VARS; ++d) {
        if (!Objective::is_symmetry_fixed(d)) {
            ++count;
        }
    }
    return count;
}

/**
 * @brief Each thread evaluates Objective's interval bounds over a subdomain
 * indexed by its thread id. Each of the free variables is evenly divided N
 * ways, so there are N^free_var_count<Objective>() total subdomains.
 *
 * @param parent Parent domain: exactly Objective::VARS interval bounds, one per variable
 * @param bounds Storage for the per-subregion objective bound
 * @param lb_out Storage for the per-subregion lower bound alone (bounds[r].lb duplicated),
 *               so the caller can cub::DeviceReduce::Max it into `max_lb` (Theorem 6.1's
 *               excess bracket, design/REFINEMENT_STUDY.md §6.1/§9.3) without a
 *               struct-field reduction iterator
 * @param ub_out Storage for the per-subregion upper bound alone, reduced with Min into `min_ub`
 * @param N Number of subdivisions per free variable
 */
template<typename T, typename Objective>
__global__ void bound_subregions_kernel(const cu::interval<T> *parent,
                                         cu::interval<T> *bounds,
                                         T *lb_out,
                                         T *ub_out,
                                         std::size_t N) {
    std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    std::size_t total_subregions = 1;
    for (std::size_t d = 0; d < free_var_count<Objective>(); ++d) {
        total_subregions *= N;
    }
    if (tid >= total_subregions) {
        return;
    }

    cu::interval<T> x[Objective::VARS];
    subregion_bounds<T, Objective>(parent, N, tid, x);

    cu::interval<T> res = Objective::eval(x);
    bounds[tid] = res;
    lb_out[tid] = res.lb;
    ub_out[tid] = res.ub;
}


template<typename T>
struct HullOp {
    __device__ cu::interval<T> operator()(const cu::interval<T> &a, const cu::interval<T> &b) const {
        return cu::hull(a, b);
    }
};

template<typename T>
struct SquareOp {
    __host__ __device__ T operator()(T x) const { return x * x; }
};

template<typename T>
struct CubeOp {
    __host__ __device__ T operator()(T x) const { return x * x * x; }
};

// Host-side interval arithmetic, used by each Objective's host_eval to
// compute the "no refinement" baseline: cuinterval's own operators
// (operations.cuh) are __device__-only (outward-rounded via intrinsic::
// *_down/_up, which have no host implementation here), so this reimplements
// the handful of operations the objectives below need directly against
// cu::interval<T>'s plain {lb, ub} fields. Unlike the library's versions,
// these use ordinary double rounding rather than outward rounding --
// adequate for comparing refinement *tightness* against the GPU result, not
// for a certified bound.
namespace host_interval
{

template<typename T>
inline cu::interval<T> add(cu::interval<T> a, cu::interval<T> b) {
    return { a.lb + b.lb, a.ub + b.ub };
}

template<typename T>
inline cu::interval<T> sub(cu::interval<T> a, cu::interval<T> b) {
    return { a.lb - b.ub, a.ub - b.lb };
}

// a - c, c a scalar constant.
template<typename T>
inline cu::interval<T> sub_scalar(cu::interval<T> a, T c) {
    return { a.lb - c, a.ub - c };
}

// c - a, c a scalar constant: decreasing in a, so the endpoints flip.
template<typename T>
inline cu::interval<T> rsub(T c, cu::interval<T> a) {
    return { c - a.ub, c - a.lb };
}

// c * a for a compile-time-known non-negative scalar c, so no sign
// case-split is needed.
template<typename T>
inline cu::interval<T> scale(T c, cu::interval<T> a) {
    return { c * a.lb, c * a.ub };
}

// -a: negation flips endpoints. Composes with scale() to cover negative
// scalars (c * a == neg(scale(-c, a)) for c < 0).
template<typename T>
inline cu::interval<T> neg(cu::interval<T> a) {
    return { -a.ub, -a.lb };
}

// x^2 over an interval that may straddle zero.
template<typename T>
inline cu::interval<T> sqr(cu::interval<T> a) {
    T lo = (a.lb <= T(0) && a.ub >= T(0)) ? T(0) : std::min(a.lb * a.lb, a.ub * a.ub);
    T hi = std::max(a.lb * a.lb, a.ub * a.ub);
    return { lo, hi };
}

// sqrt is monotonic increasing; caller guarantees a.lb >= 0 (true here since
// r2 is a sum of sqr() results).
template<typename T>
inline cu::interval<T> sqrt(cu::interval<T> a) {
    return { std::sqrt(a.lb), std::sqrt(a.ub) };
}

// exp is monotonic increasing.
template<typename T>
inline cu::interval<T> exp(cu::interval<T> a) {
    return { std::exp(a.lb), std::exp(a.ub) };
}

}  // namespace host_interval


// ---- Objectives ---------------------------------------------------------
//
// Each Objective struct is a problem definition: VARS (compile-time dim
// count), is_symmetry_fixed(d) (which dims are pinned to a point interval
// and so don't get subdivided), eval() (device interval evaluation, mirrors
// the corresponding source/fixed_examples/*.cu problem), host_eval() (host
// mirror over host_interval, the unrefined/parent-domain baseline), and
// build_parent() (the VARS-entry parent domain to subdivide).

// The morse cluster potential over POINTS 3D points
// (source/fixed_examples/graph_morse_cluster.cu, minlplib ex8_6_2). 6 dims
// fixed to break translational/rotational symmetry: point 0's x, point 0 and
// point 1's y, and point 0/1/2's z. Point 0 pinned at the origin removes
// translation; point 1 pinned onto the x-axis (y=z=0) and point 2 pinned
// into the xy-plane (z=0) removes rotation. Requires at least 3 points for
// point 2 to exist.
template<typename T>
struct MorseClusterObjective {
    static constexpr std::size_t POINTS = 4;
    static constexpr std::size_t VARS = 3 * POINTS;
    static constexpr const char *name = "morse";

    static_assert(POINTS >= 3, "symmetry-breaking needs at least 3 points");

    __host__ __device__ static constexpr bool is_symmetry_fixed(std::size_t d) {
        return d == 0 || d == POINTS || d == POINTS + 1
            || d == 2 * POINTS || d == 2 * POINTS + 1 || d == 2 * POINTS + 2;
    }

    __device__ static cu::interval<T> eval(const cu::interval<T> *x) {
        cu::interval<T> res = cu::interval<T>(-45);
        for (std::size_t i = 0; i < POINTS - 1; ++i) {
            for (std::size_t j = i + 1; j < POINTS; ++j) {
                auto dx = x[i] - x[j];
                auto dy = x[i + POINTS] - x[j + POINTS];
                auto dz = x[i + 2 * POINTS] - x[j + 2 * POINTS];
                auto r2 = sqr(dx) + sqr(dy) + sqr(dz);
                auto r = sqrt(r2);
                auto t = exp(3 * (1 - r));
                auto u = 1 - t;
                auto v = sqr(u);
                res = res + v;
            }
        }
        return res;
    }

    static cu::interval<T> host_eval(const cu::interval<T> *x) {
        namespace hi = host_interval;
        cu::interval<T> res { T(-45), T(-45) };
        for (std::size_t i = 0; i < POINTS - 1; ++i) {
            for (std::size_t j = i + 1; j < POINTS; ++j) {
                auto dx = hi::sub(x[i], x[j]);
                auto dy = hi::sub(x[i + POINTS], x[j + POINTS]);
                auto dz = hi::sub(x[i + 2 * POINTS], x[j + 2 * POINTS]);
                auto r2 = hi::add(hi::add(hi::sqr(dx), hi::sqr(dy)), hi::sqr(dz));
                auto r = hi::sqrt(r2);
                auto t = hi::exp(hi::scale(T(3), hi::rsub(T(1), r)));
                auto u = hi::rsub(T(1), t);
                auto v = hi::sqr(u);
                res = hi::add(res, v);
            }
        }
        return res;
    }

    // Unit-width sub-boxes tiling [-5, 5], skipping [-1, 0] and [0, 1] so no
    // free variable's domain contains the origin -- with every free
    // variable spanning the full [-5, 5], the pairwise-distance term for any
    // two points can always be driven to r=0 by placing both at 0, which
    // pins the objective's interval upper bound to the same "all points
    // collapse" value at every N and hides the refinement effect. Free dims
    // are assigned round-robin over this list, so dims sharing a box
    // (free_var_count<MorseClusterObjective>() > 8) can still coincide with
    // each other, but never at the origin.
    static void build_parent(std::vector<cu::interval<T>> &parent) {
        constexpr double free_domain_lb[] = { -5, -4, -3, -2, 1, 2, 3, 4 };
        constexpr std::size_t num_free_domains = sizeof(free_domain_lb) / sizeof(free_domain_lb[0]);

        parent.assign(VARS, cu::interval<T>(0.0));
        std::size_t free_idx = 0;
        for (std::size_t d = 0; d < VARS; ++d) {
            if (is_symmetry_fixed(d)) {
                continue;
            }
            double lb = free_domain_lb[free_idx % num_free_domains];
            parent[d] = cu::interval<T>(lb, lb + 1.0);
            ++free_idx;
        }
    }
};

// The Rosenbrock valley function (source/fixed_examples/graph_rosenbrock.cu,
// minlplib prob09 generalised to VARS > 2), restricted to a small VARS so
// N^free_vars stays tractable -- the full example uses 100 dims. No
// symmetry to fix: every dim is free and uses its natural [-30, 30] bound.
template<typename T>
struct RosenbrockObjective {
    static constexpr std::size_t VARS = 6;
    static constexpr const char *name = "rosenbrock";

    __host__ __device__ static constexpr bool is_symmetry_fixed(std::size_t) { return false; }

    __device__ static cu::interval<T> eval(const cu::interval<T> *x) {
        cu::interval<T> res = cu::interval<T>(0);
        for (std::size_t i = 0; i + 1 < VARS; ++i) {
            auto a = sqr(x[i]) - x[i + 1];
            auto b = x[i] - 1;
            res = res + (100 * sqr(a) + sqr(b));
        }
        return res;
    }

    static cu::interval<T> host_eval(const cu::interval<T> *x) {
        namespace hi = host_interval;
        cu::interval<T> res { T(0), T(0) };
        for (std::size_t i = 0; i + 1 < VARS; ++i) {
            auto a = hi::sub(hi::sqr(x[i]), x[i + 1]);
            auto b = hi::sub_scalar(x[i], T(1));
            auto term = hi::add(hi::scale(T(100), hi::sqr(a)), hi::sqr(b));
            res = hi::add(res, term);
        }
        return res;
    }

    static void build_parent(std::vector<cu::interval<T>> &parent) {
        parent.assign(VARS, cu::interval<T>(-30.0, 30.0));
    }
};

// minlplib ex2_1_2 (source/fixed_examples/qps.cu's problem_212), objective
// only -- this pipeline studies unconstrained interval bound refinement, so
// ex2_1_2's two linear constraints are not applied here. x0..x3 are the
// [0, 1] variables, x5 is the [0, 20] one; natural per-variable bounds, no
// symmetry to fix.
template<typename T>
struct Ex212Objective {
    static constexpr std::size_t VARS = 6;
    static constexpr const char *name = "ex212";

    __host__ __device__ static constexpr bool is_symmetry_fixed(std::size_t) { return false; }

    __device__ static cu::interval<T> eval(const cu::interval<T> *x) {
        cu::interval<T> res = -0.5
            * (sqr(x[0]) + sqr(x[1]) + sqr(x[2]) + sqr(x[3]) + sqr(x[4]));
        res = res - 10.5 * x[0] - 7.5 * x[1] - 3.5 * x[2] - 2.5 * x[3] - 1.5 * x[4] - 10 * x[5];
        return res;
    }

    static cu::interval<T> host_eval(const cu::interval<T> *x) {
        namespace hi = host_interval;
        auto sumsq = hi::add(
            hi::add(hi::add(hi::add(hi::sqr(x[0]), hi::sqr(x[1])), hi::sqr(x[2])), hi::sqr(x[3])),
            hi::sqr(x[4]));
        cu::interval<T> res = hi::neg(hi::scale(T(0.5), sumsq));
        res = hi::sub(res, hi::scale(T(10.5), x[0]));
        res = hi::sub(res, hi::scale(T(7.5), x[1]));
        res = hi::sub(res, hi::scale(T(3.5), x[2]));
        res = hi::sub(res, hi::scale(T(2.5), x[3]));
        res = hi::sub(res, hi::scale(T(1.5), x[4]));
        res = hi::sub(res, hi::scale(T(10.0), x[5]));
        return res;
    }

    static void build_parent(std::vector<cu::interval<T>> &parent) {
        parent.assign(VARS, cu::interval<T>(0.0, 1.0));
        parent[5] = cu::interval<T>(0.0, 20.0);
    }
};


// ---- Study driver ---------------------------------------------------------

template<typename T, typename Objective>
void run_study(const std::filesystem::path &results_dir) {
    constexpr std::size_t n_values[] = { 1, 2, 4, 8, 16, 32 };

    std::filesystem::create_directories(results_dir);

    std::vector<cu::interval<T>> h_parent;
    Objective::build_parent(h_parent);

    cu::interval<T> *d_parent = nullptr;
    cuda_check(cudaMalloc(&d_parent, Objective::VARS * sizeof(cu::interval<T>)), "cudaMalloc d_parent");
    cuda_check(cudaMemcpy(d_parent, h_parent.data(), Objective::VARS * sizeof(cu::interval<T>),
                           cudaMemcpyHostToDevice),
               "cudaMemcpy d_parent");

    cu::interval<T> unrefined = Objective::host_eval(h_parent.data());

    constexpr std::size_t free_vars = free_var_count<Objective>();
    double avg_free_width = 0;
    for (std::size_t d = 0; d < Objective::VARS; ++d) {
        if (!Objective::is_symmetry_fixed(d)) {
            avg_free_width += (h_parent[d].ub - h_parent[d].lb);
        }
    }
    avg_free_width /= static_cast<double>(free_vars);

    std::printf("problem:    %-10s VARS=%-4zu free_vars=%zu\n", Objective::name, Objective::VARS, free_vars);
    std::printf("unrefined (parent domain):   [% .6f, % .6f]  width=%.6f\n",
                unrefined.lb, unrefined.ub, unrefined.ub - unrefined.lb);
    std::printf("\n");

    for (std::size_t N : n_values) {
        std::size_t total_subregions = 1;
        for (std::size_t d = 0; d < free_vars; ++d) {
            total_subregions *= N;
        }
        if (total_subregions > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr,
                          "N=%zu, free_vars=%zu -> N^free_vars = %zu subregions exceeds CUB's int item count, skipping\n",
                          N, free_vars, total_subregions);
            continue;
        }

        cu::interval<T> *d_bounds = nullptr;
        cu::interval<T> *d_refined = nullptr;
        T *d_lb = nullptr;
        T *d_ub = nullptr;
        T *d_min_ub = nullptr;
        T *d_max_lb = nullptr;
        T *d_sum_lb = nullptr;
        T *d_sumsq_lb = nullptr;
        T *d_sumcube_lb = nullptr;
        cuda_check(cudaMalloc(&d_bounds, total_subregions * sizeof(cu::interval<T>)), "cudaMalloc d_bounds");
        cuda_check(cudaMalloc(&d_refined, sizeof(cu::interval<T>)), "cudaMalloc d_refined");
        cuda_check(cudaMalloc(&d_lb, total_subregions * sizeof(T)), "cudaMalloc d_lb");
        cuda_check(cudaMalloc(&d_ub, total_subregions * sizeof(T)), "cudaMalloc d_ub");
        cuda_check(cudaMalloc(&d_min_ub, sizeof(T)), "cudaMalloc d_min_ub");
        cuda_check(cudaMalloc(&d_max_lb, sizeof(T)), "cudaMalloc d_max_lb");
        cuda_check(cudaMalloc(&d_sum_lb, sizeof(T)), "cudaMalloc d_sum_lb");
        cuda_check(cudaMalloc(&d_sumsq_lb, sizeof(T)), "cudaMalloc d_sumsq_lb");
        cuda_check(cudaMalloc(&d_sumcube_lb, sizeof(T)), "cudaMalloc d_sumcube_lb");

        constexpr int block = 256;
        int grid = static_cast<int>((total_subregions + block - 1) / block);
        bound_subregions_kernel<T, Objective><<<grid, block>>>(d_parent, d_bounds, d_lb, d_ub, N);
        cuda_check(cudaGetLastError(), "bound_subregions_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "bound_subregions_kernel sync");

        // Raw per-subregion lb/ub, for offline distribution plotting
        // (plot_distributions.py). Copied now, before d_ub is overwritten
        // in place by the median sort below.
        {
            std::vector<double> h_lb(total_subregions);
            std::vector<double> h_ub(total_subregions);
            cuda_check(cudaMemcpy(h_lb.data(), d_lb, total_subregions * sizeof(T), cudaMemcpyDeviceToHost),
                       "cudaMemcpy h_lb");
            cuda_check(cudaMemcpy(h_ub.data(), d_ub, total_subregions * sizeof(T), cudaMemcpyDeviceToHost),
                       "cudaMemcpy h_ub");
            write_f64_array(results_dir / ("N" + std::to_string(N) + "_lb.f64"), h_lb);
            write_f64_array(results_dir / ("N" + std::to_string(N) + "_ub.f64"), h_ub);
        }

        HullOp<T> hull_op;
        // Identity element for hull: cu::empty<T>() (operations.cuh) is
        // device-only, so its {+inf, -inf} definition is reproduced here.
        cu::interval<T> hull_identity {
            std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity()
        };

        // Raw-moment sums of the per-subregion lower bounds (d_lb), taken via
        // transform iterators so no extra N-sized buffer is needed: mean/std/
        // skewness are recovered from S1=sum(x), S2=sum(x^2), S3=sum(x^3) on
        // the host below (population moments, since this is the full set of
        // subregions, not a sample of it).
        auto sq_lb_it = thrust::make_transform_iterator(d_lb, SquareOp<T>{});
        auto cube_lb_it = thrust::make_transform_iterator(d_lb, CubeOp<T>{});

        // Several independent reductions plus a sort of d_lb (median, via
        // radix sort into the now-unused d_ub buffer) share one temp-storage
        // query/alloc cycle (cub sizes them near-identically for same-size
        // inputs; querying/allocating the max of all of them keeps this simple).
        void *d_temp = nullptr;
        std::size_t temp_bytes = 0, hull_bytes = 0, min_bytes = 0, max_bytes = 0;
        std::size_t sum_bytes = 0, sumsq_bytes = 0, sumcube_bytes = 0, sort_bytes = 0;
        cuda_check(cub::DeviceReduce::Reduce(nullptr, hull_bytes, d_bounds, d_refined,
                                              static_cast<int>(total_subregions), hull_op, hull_identity),
                   "cub::DeviceReduce::Reduce (size query)");
        cuda_check(cub::DeviceReduce::Min(nullptr, min_bytes, d_ub, d_min_ub, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Min (size query)");
        cuda_check(cub::DeviceReduce::Max(nullptr, max_bytes, d_lb, d_max_lb, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Max (size query)");
        cuda_check(cub::DeviceReduce::Sum(nullptr, sum_bytes, d_lb, d_sum_lb, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Sum (size query)");
        cuda_check(cub::DeviceReduce::Sum(nullptr, sumsq_bytes, sq_lb_it, d_sumsq_lb,
                                           static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Sum sq (size query)");
        cuda_check(cub::DeviceReduce::Sum(nullptr, sumcube_bytes, cube_lb_it, d_sumcube_lb,
                                           static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Sum cube (size query)");
        cuda_check(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, d_lb, d_ub,
                                                    static_cast<int>(total_subregions)),
                   "cub::DeviceRadixSort::SortKeys (size query)");
        temp_bytes = std::max({ hull_bytes, min_bytes, max_bytes, sum_bytes, sumsq_bytes, sumcube_bytes, sort_bytes });
        cuda_check(cudaMalloc(&d_temp, temp_bytes), "cudaMalloc d_temp");

        cuda_check(cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_bounds, d_refined,
                                              static_cast<int>(total_subregions), hull_op, hull_identity),
                   "cub::DeviceReduce::Reduce");
        cuda_check(cub::DeviceReduce::Min(d_temp, temp_bytes, d_ub, d_min_ub, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Min");
        cuda_check(cub::DeviceReduce::Max(d_temp, temp_bytes, d_lb, d_max_lb, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Max");
        cuda_check(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_lb, d_sum_lb, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Sum");
        cuda_check(cub::DeviceReduce::Sum(d_temp, temp_bytes, sq_lb_it, d_sumsq_lb,
                                           static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Sum sq");
        cuda_check(cub::DeviceReduce::Sum(d_temp, temp_bytes, cube_lb_it, d_sumcube_lb,
                                           static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Sum cube");
        // Sorts d_lb (no longer needed unsorted -- max_lb above already
        // captured its reduction) into d_ub (likewise done with its own
        // min reduction), so the median can be read off without an extra
        // N-sized allocation.
        cuda_check(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_lb, d_ub,
                                                    static_cast<int>(total_subregions)),
                   "cub::DeviceRadixSort::SortKeys");

        cu::interval<T> refined {};
        T min_ub = 0, max_lb = 0;
        cuda_check(cudaMemcpy(&refined, d_refined, sizeof(refined), cudaMemcpyDeviceToHost),
                   "cudaMemcpy refined");
        cuda_check(cudaMemcpy(&min_ub, d_min_ub, sizeof(min_ub), cudaMemcpyDeviceToHost), "cudaMemcpy min_ub");
        cuda_check(cudaMemcpy(&max_lb, d_max_lb, sizeof(max_lb), cudaMemcpyDeviceToHost), "cudaMemcpy max_lb");

        T sum_lb = 0, sumsq_lb = 0, sumcube_lb = 0;
        cuda_check(cudaMemcpy(&sum_lb, d_sum_lb, sizeof(sum_lb), cudaMemcpyDeviceToHost), "cudaMemcpy sum_lb");
        cuda_check(cudaMemcpy(&sumsq_lb, d_sumsq_lb, sizeof(sumsq_lb), cudaMemcpyDeviceToHost), "cudaMemcpy sumsq_lb");
        cuda_check(cudaMemcpy(&sumcube_lb, d_sumcube_lb, sizeof(sumcube_lb), cudaMemcpyDeviceToHost),
                   "cudaMemcpy sumcube_lb");

        double n = static_cast<double>(total_subregions);
        double lb_mean = sum_lb / n;
        double lb_variance = sumsq_lb / n - lb_mean * lb_mean;
        double lb_stddev = std::sqrt(std::max(lb_variance, 0.0));
        double lb_m3 = sumcube_lb / n - 3.0 * lb_mean * (sumsq_lb / n) + 2.0 * lb_mean * lb_mean * lb_mean;
        double lb_skewness = (lb_stddev > 0.0) ? lb_m3 / (lb_stddev * lb_stddev * lb_stddev) : 0.0;

        // d_ub now holds d_lb sorted ascending (the SortKeys call above).
        double lb_median;
        if (total_subregions % 2 == 1) {
            T mid;
            cuda_check(cudaMemcpy(&mid, d_ub + total_subregions / 2, sizeof(mid), cudaMemcpyDeviceToHost),
                       "cudaMemcpy lb_median");
            lb_median = mid;
        } else {
            T lo, hi;
            cuda_check(cudaMemcpy(&lo, d_ub + total_subregions / 2 - 1, sizeof(lo), cudaMemcpyDeviceToHost),
                       "cudaMemcpy lb_median lo");
            cuda_check(cudaMemcpy(&hi, d_ub + total_subregions / 2, sizeof(hi), cudaMemcpyDeviceToHost),
                       "cudaMemcpy lb_median hi");
            lb_median = (static_cast<double>(lo) + static_cast<double>(hi)) / 2.0;
        }

        // Theorem 6.1's excess bracket (design/REFINEMENT_STUDY.md §6.1/§9.3):
        // min_r ub_r >= max f >= L_N and U_N >= min f... precisely, min_ub and
        // max_lb sandwich the true range [min f, max f] from the inside, so
        // (min_ub - L_N) + (U_N - max_lb) is a sound upper bound on
        // excess(N) = W_N - range(f), computable without knowing range(f). It
        // is this excess, not the raw hull width W_N = U_N - L_N (which floors
        // out at range(f) rather than 0), that is expected to shrink linearly
        // (per free dimension) as N grows.
        double excess_bound = (static_cast<double>(min_ub) - refined.lb) + (refined.ub - static_cast<double>(max_lb));

        // Every free variable's domain need not be the same width (e.g.
        // Ex212Objective mixes [0,1] and [0,20] free dims), so this is the
        // *average* free-dim width under an N-way split -- the quantity
        // Theorem 6.1 predicts excess(N) is (per-dimension) linear in when
        // widths are uniform.
        double avg_subdomain_width = avg_free_width / static_cast<double>(N);

        std::printf("refinement: N=%-4zu avg_subdomain_width=%-10.6f subregions=N^free_vars=%zu\n",
                    N, avg_subdomain_width, total_subregions);
        std::printf("  refined     (N-way hull):    [% .6f, % .6f]  width=%.6f\n",
                    refined.lb, refined.ub, refined.ub - refined.lb);
        std::printf("  excess bound (Theorem 6.1):  min_ub=% .6f max_lb=% .6f  excess<=%.6f\n",
                    static_cast<double>(min_ub), static_cast<double>(max_lb), excess_bound);
        std::printf("  subdomain lower-bound distribution:  mean=% .6f median=% .6f stddev=%.6f skewness=% .6f\n",
                    lb_mean, lb_median, lb_stddev, lb_skewness);

        cudaFree(d_bounds);
        cudaFree(d_refined);
        cudaFree(d_lb);
        cudaFree(d_ub);
        cudaFree(d_min_ub);
        cudaFree(d_max_lb);
        cudaFree(d_sum_lb);
        cudaFree(d_sumsq_lb);
        cudaFree(d_sumcube_lb);
        cudaFree(d_temp);
    }

    cudaFree(d_parent);
}

auto main(int argc, char **argv) -> int {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <morse|rosenbrock|ex212> [results_dir]\n", argv[0]);
        return 1;
    }
    std::string problem = argv[1];
    std::filesystem::path results_dir = (argc > 2)
        ? std::filesystem::path(argv[2])
        : std::filesystem::path("study/variable_subdomain/results") / problem;

    if (problem == "morse") {
        run_study<double, MorseClusterObjective<double>>(results_dir);
    } else if (problem == "rosenbrock") {
        run_study<double, RosenbrockObjective<double>>(results_dir);
    } else if (problem == "ex212") {
        run_study<double, Ex212Objective<double>>(results_dir);
    } else {
        std::fprintf(stderr, "unknown problem '%s' (expected morse|rosenbrock|ex212)\n", problem.c_str());
        return 1;
    }

    return 0;
}
