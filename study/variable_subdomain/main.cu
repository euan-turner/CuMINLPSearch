#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <cub/cub.cuh>
#include <cuinterval/cuinterval.h>

// For 5 and 10 points (15 and 30 variables), a range of subdomains/widths, and a range of original box widths,
// record the bounds that interval refinement produces.

// Requires:
// 1. A kernel to determine bounds over a subdomain, given a subregion index
// 2. Min and Max reduction over those bounds to determine the refinement


namespace
{
constexpr std::size_t POINTS = 5;
constexpr std::size_t DIMS = 3 * POINTS;

void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}
}


// The 6 dims fixed to break translational/rotational symmetry (mirrors
// graph_morse_cluster.cu's x1, x11, x12, x21, x22, x23): point 0's x, point
// 0 and point 1's y, and point 0/1/2's z. Point 0 pinned at the origin
// removes translation; point 1 pinned onto the x-axis (y=z=0) and point 2
// pinned into the xz=0 half-plane... actually the xy-plane (z=0) removes
// rotation. Requires at least 3 points for point 2 to exist.
template<std::size_t VARS>
__host__ __device__ constexpr bool is_symmetry_fixed(std::size_t d) {
    constexpr std::size_t POINTS = VARS / 3;
    static_assert(POINTS >= 3, "symmetry-breaking needs at least 3 points");
    return d == 0 || d == POINTS || d == POINTS + 1
        || d == 2 * POINTS || d == 2 * POINTS + 1 || d == 2 * POINTS + 2;
}

template<std::size_t VARS>
__host__ __device__ constexpr std::size_t free_var_count() {
    std::size_t count = 0;
    for (std::size_t d = 0; d < VARS; ++d) {
        if (!is_symmetry_fixed<VARS>(d)) {
            ++count;
        }
    }
    return count;
}

/**
 * @brief Maps a thread id (linear subregion index) to per-dimension bounds,
 * under a uniform N-way subdivision of every free variable's bounds. The 6
 * symmetry-fixed dims (is_symmetry_fixed) are copied straight from `parent`
 * and consume no digit of `sidx` -- they're already point intervals, so
 * subdividing them would only inflate the subregion count by N^6 for
 * identical copies.
 *
 * @tparam T
 * @tparam VARS Problem dimension (compile-time, so is_symmetry_fixed can key off it)
 * @param parent Parent domain bounds, VARS entries
 * @param N Subdivision per free variable
 * @param sidx Subregion index, in [0, N^free_var_count<VARS>())
 * @param out VARS entries
 */
template<typename T, std::size_t VARS>
__host__ __device__ inline void subregion_bounds(const cu::interval<T> *parent,
                                                   std::size_t N,
                                                   std::size_t sidx,
                                                   cu::interval<T> *out) {
    for (std::size_t d = 0; d < VARS; ++d) {
        if (is_symmetry_fixed<VARS>(d)) {
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

/**
 * @brief Each thread evaluates the morse cluster energy objective's interval
 * bounds over a subdomain indexed by its thread id. Each of the VARS - 6
 * free variables is evenly divided N ways, so there are
 * N^free_var_count<VARS>() total subdomains.
 *
 * @param parent Parent domain: exactly VARS interval bounds, one per variable
 * @param bounds Storage for the per-subregion objective bound
 * @param lb_out Storage for the per-subregion lower bound alone (bounds[r].lb duplicated),
 *               so the caller can cub::DeviceReduce::Max it into `max_lb` (Theorem 6.1's
 *               excess bracket, design/REFINEMENT_STUDY.md §6.1/§9.3) without a
 *               struct-field reduction iterator
 * @param ub_out Storage for the per-subregion upper bound alone, reduced with Min into `min_ub`
 * @param N Number of subdivisions per free variable
 */
template<typename T, std::size_t VARS>
__global__ void bound_subregions_kernel(const cu::interval<T> *parent,
                                         cu::interval<T> *bounds,
                                         T *lb_out,
                                         T *ub_out,
                                         std::size_t N) {
    std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr std::size_t POINTS = VARS / 3;

    std::size_t total_subregions = 1;
    for (std::size_t d = 0; d < free_var_count<VARS>(); ++d) {
        total_subregions *= N;
    }
    if (tid >= total_subregions) {
        return;
    }

    cu::interval<T> x[VARS];
    subregion_bounds<T, VARS>(parent, N, tid, x);

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

// Host-side interval arithmetic for the "no refinement" baseline: cuinterval's
// own operators (operations.cuh) are __device__-only (outward-rounded via
// intrinsic::*_down/_up, which have no host implementation here), so this
// reimplements the handful of operations the morse objective needs directly
// against cu::interval<T>'s plain {lb, ub} fields. Unlike the library's
// versions, these use ordinary double rounding rather than outward rounding
// -- adequate for comparing refinement *tightness* against the GPU result,
// not for a certified bound.
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

// c - a, c a scalar constant: decreasing in a, so the endpoints flip.
template<typename T>
inline cu::interval<T> rsub(T c, cu::interval<T> a) {
    return { c - a.ub, c - a.lb };
}

// c * a for a compile-time-known non-negative scalar c (the objective only
// ever scales by +3), so no sign case-split is needed.
template<typename T>
inline cu::interval<T> scale(T c, cu::interval<T> a) {
    return { c * a.lb, c * a.ub };
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

// CPU mirror of bound_subregions_kernel's per-thread objective loop, applied
// once to the whole (unsubdivided) parent domain -- the "no refinement"
// baseline the GPU's N-way hull is compared against. Must be kept in sync
// with the device loop above by hand; the two can't share code since one
// runs on cu::interval<T>'s device-only operators and the other on
// host_interval's host-only ones.
template<typename T, std::size_t VARS>
cu::interval<T> host_morse_energy_bound(const cu::interval<T> *x) {
    constexpr std::size_t POINTS = VARS / 3;
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

auto main() -> int {
    constexpr std::size_t n_values[] = { 1, 2, 4, 8, 16, 32 };

    // Unit-width sub-boxes tiling [-5, 5], skipping [-1, 0] and [0, 1] so no
    // free variable's domain contains the origin -- with every free
    // variable spanning the full [-5, 5], the pairwise-distance term for any
    // two points can always be driven to r=0 by placing both at 0, which
    // pins the objective's interval upper bound to the same "all points
    // collapse" value at every N and hides the refinement effect. Free
    // dims are assigned round-robin over this list, so dims sharing a box
    // (free_var_count<DIMS>() > 8) can still coincide with each other, but
    // never at the origin.
    constexpr double free_domain_lb[] = { -5, -4, -3, -2, 1, 2, 3, 4 };
    constexpr std::size_t num_free_domains = sizeof(free_domain_lb) / sizeof(free_domain_lb[0]);

    std::vector<cu::interval<double>> h_parent(DIMS);
    std::size_t free_idx = 0;
    for (std::size_t d = 0; d < DIMS; ++d) {
        if (is_symmetry_fixed<DIMS>(d)) {
            h_parent[d] = cu::interval<double>(0.0);
        } else {
            double lb = free_domain_lb[free_idx % num_free_domains];
            h_parent[d] = cu::interval<double>(lb, lb + 1.0);
            ++free_idx;
        }
    }

    cu::interval<double> *d_parent = nullptr;
    cuda_check(cudaMalloc(&d_parent, DIMS * sizeof(cu::interval<double>)), "cudaMalloc d_parent");
    cuda_check(cudaMemcpy(d_parent, h_parent.data(), DIMS * sizeof(cu::interval<double>),
                           cudaMemcpyHostToDevice),
               "cudaMemcpy d_parent");

    cu::interval<double> unrefined = host_morse_energy_bound<double, DIMS>(h_parent.data());

    std::printf("problem:    POINTS=%-4zu DIMS=%-4zu free_vars=%zu\n", POINTS, DIMS, free_var_count<DIMS>());
    std::printf("unrefined (parent domain):   [% .6f, % .6f]  width=%.6f\n",
                unrefined.lb, unrefined.ub, unrefined.ub - unrefined.lb);
    std::printf("\n");

    for (std::size_t N : n_values) {
        std::size_t total_subregions = 1;
        for (std::size_t d = 0; d < free_var_count<DIMS>(); ++d) {
            total_subregions *= N;
        }
        if (total_subregions > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr,
                          "N=%zu, DIMS=%zu -> N^DIMS = %zu subregions exceeds CUB's int item count, skipping\n",
                          N, DIMS, total_subregions);
            continue;
        }

        cu::interval<double> *d_bounds = nullptr;
        cu::interval<double> *d_refined = nullptr;
        double *d_lb = nullptr;
        double *d_ub = nullptr;
        double *d_min_ub = nullptr;
        double *d_max_lb = nullptr;
        cuda_check(cudaMalloc(&d_bounds, total_subregions * sizeof(cu::interval<double>)), "cudaMalloc d_bounds");
        cuda_check(cudaMalloc(&d_refined, sizeof(cu::interval<double>)), "cudaMalloc d_refined");
        cuda_check(cudaMalloc(&d_lb, total_subregions * sizeof(double)), "cudaMalloc d_lb");
        cuda_check(cudaMalloc(&d_ub, total_subregions * sizeof(double)), "cudaMalloc d_ub");
        cuda_check(cudaMalloc(&d_min_ub, sizeof(double)), "cudaMalloc d_min_ub");
        cuda_check(cudaMalloc(&d_max_lb, sizeof(double)), "cudaMalloc d_max_lb");

        constexpr int block = 256;
        int grid = static_cast<int>((total_subregions + block - 1) / block);
        bound_subregions_kernel<double, DIMS><<<grid, block>>>(d_parent, d_bounds, d_lb, d_ub, N);
        cuda_check(cudaGetLastError(), "bound_subregions_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "bound_subregions_kernel sync");

        HullOp<double> hull_op;
        // Identity element for hull: cu::empty<T>() (operations.cuh) is
        // device-only, so its {+inf, -inf} definition is reproduced here.
        cu::interval<double> hull_identity {
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()
        };

        // Three independent reductions share one temp-storage query/alloc cycle
        // (cub::DeviceReduce sizes them near-identically for same-size inputs;
        // querying/allocating the max of the three keeps this simple).
        void *d_temp = nullptr;
        std::size_t temp_bytes = 0, hull_bytes = 0, min_bytes = 0, max_bytes = 0;
        cuda_check(cub::DeviceReduce::Reduce(nullptr, hull_bytes, d_bounds, d_refined,
                                              static_cast<int>(total_subregions), hull_op, hull_identity),
                   "cub::DeviceReduce::Reduce (size query)");
        cuda_check(cub::DeviceReduce::Min(nullptr, min_bytes, d_ub, d_min_ub, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Min (size query)");
        cuda_check(cub::DeviceReduce::Max(nullptr, max_bytes, d_lb, d_max_lb, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Max (size query)");
        temp_bytes = std::max({ hull_bytes, min_bytes, max_bytes });
        cuda_check(cudaMalloc(&d_temp, temp_bytes), "cudaMalloc d_temp");

        cuda_check(cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_bounds, d_refined,
                                              static_cast<int>(total_subregions), hull_op, hull_identity),
                   "cub::DeviceReduce::Reduce");
        cuda_check(cub::DeviceReduce::Min(d_temp, temp_bytes, d_ub, d_min_ub, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Min");
        cuda_check(cub::DeviceReduce::Max(d_temp, temp_bytes, d_lb, d_max_lb, static_cast<int>(total_subregions)),
                   "cub::DeviceReduce::Max");

        cu::interval<double> refined {};
        double min_ub = 0, max_lb = 0;
        cuda_check(cudaMemcpy(&refined, d_refined, sizeof(refined), cudaMemcpyDeviceToHost),
                   "cudaMemcpy refined");
        cuda_check(cudaMemcpy(&min_ub, d_min_ub, sizeof(min_ub), cudaMemcpyDeviceToHost), "cudaMemcpy min_ub");
        cuda_check(cudaMemcpy(&max_lb, d_max_lb, sizeof(max_lb), cudaMemcpyDeviceToHost), "cudaMemcpy max_lb");

        // Theorem 6.1's excess bracket (design/REFINEMENT_STUDY.md §6.1/§9.3):
        // min_r ub_r >= max f >= L_N and U_N >= min f... precisely, min_ub and
        // max_lb sandwich the true range [min f, max f] from the inside, so
        // (min_ub - L_N) + (U_N - max_lb) is a sound upper bound on
        // excess(N) = W_N - range(f), computable without knowing range(f). It
        // is this excess, not the raw hull width W_N = U_N - L_N (which floors
        // out at range(f) rather than 0), that is expected to shrink linearly
        // (per free dimension) as N grows.
        double excess_bound = (min_ub - refined.lb) + (refined.ub - max_lb);

        // Every free variable's domain is unit-width (free_domain_lb above), so
        // an N-way split makes each subdomain 1/N wide on every free axis --
        // the quantity Theorem 6.1 predicts excess(N) is linear in.
        double subdomain_width = 1.0 / static_cast<double>(N);

        std::printf("refinement: N=%-4zu subdomain_width=%-10.6f subregions=N^free_vars=%zu\n",
                    N, subdomain_width, total_subregions);
        std::printf("  refined     (N-way hull):    [% .6f, % .6f]  width=%.6f\n",
                    refined.lb, refined.ub, refined.ub - refined.lb);
        std::printf("  excess bound (Theorem 6.1):  min_ub=% .6f max_lb=% .6f  excess<=%.6f\n",
                    min_ub, max_lb, excess_bound);

        cudaFree(d_bounds);
        cudaFree(d_refined);
        cudaFree(d_lb);
        cudaFree(d_ub);
        cudaFree(d_min_ub);
        cudaFree(d_max_lb);
        cudaFree(d_temp);
    }

    cudaFree(d_parent);

    return 0;
}
