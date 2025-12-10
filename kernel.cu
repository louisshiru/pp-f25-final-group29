#include "dataloader.h"
#include "kernel.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <vector>
#include <utility>

#define GEN_COLLECT_INTERVAL 100

// Maximum cities supported for certain stack-allocated per-thread buffers.
#define MAX_CITIES 4096

// ======================== Cuda implementation ========================

// Global persistent GA config used by CUDA kernels.
static GA2OptConfig g_cfg = {};

// Device kernel: reduce partial sums to total fitness (single thread)
__global__ void ga2opt_reduce_total_fitness_kernel(const double* partial_sums, int n_blocks, double* total_fitness) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double total = 0.0;
    for (int i = 0; i < n_blocks; ++i) {
        total += partial_sums[i];
    }
    *total_fitness = total;
}

// Helper to index into the precomputed distance matrix.
__device__ __forceinline__ double d_dist_lookup(const GA2OptConfig& cfg, int i, int j) {
    const double* mat = cfg.d_dist_matrix;
    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(cfg.n_cities) + static_cast<size_t>(j);
    return __ldg(&mat[idx]);
}

__device__ double d_total_distance(const GA2OptConfig& cfg, const int* route, int n) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        int a = route[i];
        int b = route[(i + 1) % n];
        total += d_dist_lookup(cfg, a, b);
    }
    return total;
}

__device__ void d_reverse_segment(int* route, int i, int k) {
    while (i < k) {
        int temp = route[i];
        route[i] = route[k];
        route[k] = temp;
        ++i;
        --k;
    }
}

__device__ double rand_real_device(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (double)(state & 0xFFFFFF) / (double)0xFFFFFF;
}

__device__ int rand_int_device(unsigned int& state, int min, int max) {
    unsigned int r = state;
    return min + (int)(r % (unsigned int)(max - min + 1));
}

__device__ int roulette_device(const int n_population, const double* probs, unsigned int& state) {
    double r = rand_real_device(state);
    double cum_prob = 0.0;
    for (int i = 0; i < n_population; ++i) {
        cum_prob += probs[i];
        if (r <= cum_prob) {
            return i;
        }
    }
    return n_population - 1;
}

__device__ void mutate_device(const GA2OptConfig& cfg, int* chrom, unsigned int& state) {
    if (rand_real_device(state) < cfg.mutation_rate) {
        int i = rand_int_device(state, 0, cfg.n_cities - 1);
        int j = rand_int_device(state, 0, cfg.n_cities - 1);
        int temp = chrom[i];
        chrom[i] = chrom[j];
        chrom[j] = temp;
    }
}

__device__ void two_opt_device(const GA2OptConfig& cfg, int* route, int max_passes) {
    int n = cfg.n_cities;
    if (n < 4) return;
    bool improved = true;
    int passes = 0;
    // Build inverse mapping: pos[city_id] -> position in route
    int pos[MAX_CITIES];
    for (int idx = 0; idx < n; ++idx) {
        pos[route[idx]] = idx;
    }

    while (improved && passes < max_passes) {
        improved = false;

        // If no neighbor table provided, fallback to full scan
        if (cfg.n_neighbors <= 0 || cfg.d_neighbors == nullptr) {
            for (int i = 1; i < n - 1 && !improved; ++i) {
                for (int k = i + 1; k < n && !improved; ++k) {
                    int next = (k + 1 == n) ? 0 : (k + 1);
                    double delta =
                        d_dist_lookup(cfg, route[i - 1], route[k]) +
                        d_dist_lookup(cfg, route[i],     route[next]) -
                        d_dist_lookup(cfg, route[i - 1], route[i]) -
                        d_dist_lookup(cfg, route[k],     route[next]);
                    if (delta < -1e-9) {
                        d_reverse_segment(route, i, k);
                        improved = true;
                    }
                }
            }
        } else {
            int nn = cfg.n_neighbors;
            for (int i = 1; i < n - 1 && !improved; ++i) {
                int city_a = route[i];
                int base = city_a * nn;
                for (int m = 0; m < nn && !improved; ++m) {
                    int city_b = cfg.d_neighbors[base + m];
                    int k = pos[city_b];
                    if (k <= i) continue;
                    int next = (k + 1 == n) ? 0 : (k + 1);
                    double delta =
                        d_dist_lookup(cfg, route[i - 1], route[k]) +
                        d_dist_lookup(cfg, route[i],     route[next]) -
                        d_dist_lookup(cfg, route[i - 1], route[i]) -
                        d_dist_lookup(cfg, route[k],     route[next]);
                    if (delta < -1e-9) {
                        d_reverse_segment(route, i, k);
                        improved = true;
                    }
                }
            }
        }
        ++passes;
    }
}

__device__ void crossover_device(const GA2OptConfig& cfg, unsigned int& state, const int* p1, const int* p2, int* c1, int* c2) {
    int cut1 = rand_int_device(state, 0, cfg.n_cities - 1);
    int cut2 = rand_int_device(state, 0, cfg.n_cities - 1);
    if (cut1 > cut2) { int t = cut1; cut1 = cut2; cut2 = t; }
    if (cfg.n_cities > MAX_CITIES) return;
    bool used[MAX_CITIES];
    auto create_child = [&](const int* pa, const int* pb, int* child) {
        if (!child) return;
        for (int i = 0; i < cfg.n_cities; ++i) {
            child[i] = -1;
            used[i] = false;
        }
        for (int i = cut1; i <= cut2; ++i) {
            child[i] = pa[i];
            used[pa[i]] = true;
        }
        int idx_b = (cut2 + 1) % cfg.n_cities;
        int idx_child = (cut2 + 1) % cfg.n_cities;
        for (int i = 0; i < cfg.n_cities; ++i) {
            int candidate = pb[idx_b];
            if (!used[candidate]) {
                child[idx_child] = candidate;
                used[candidate] = true;
                idx_child = (idx_child + 1) % cfg.n_cities;
            }
            idx_b = (idx_b + 1) % cfg.n_cities;
        }
    };
    create_child(p1, p2, c1);
    create_child(p2, p1, c2);
};

__device__ void fitness_device(const GA2OptConfig& cfg, const int* chrom, double* out_distance) {
    *out_distance = d_total_distance(cfg, chrom, cfg.n_cities);
}

__device__ int find_best_individual_device(const GA2OptConfig& cfg, const double* distances) {
    int best_idx = 0;
    for (int i = 1; i < cfg.n_population; ++i) {
        if (distances[i] < distances[best_idx]) {
            best_idx = i;
        }
    }
    return best_idx;
}

__device__ void copy_individual_device(const GA2OptConfig& cfg, const int* src, int* dest) {
    for (int i = 0; i < cfg.n_cities; ++i) {
        dest[i] = src[i];
    }
}

// Single-thread kernel to find the best individual on device
__global__ void ga2opt_find_best_kernel(GA2OptConfig cfg, const double* d_dist, int* d_best_idx) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int best = 0;
    double best_val = d_dist[0];
    for (int i = 1; i < cfg.n_population; ++i) {
        double v = d_dist[i];
        if (v < best_val) {
            best_val = v;
            best = i;
        }
    }
    *d_best_idx = best;
}

// Compute fitness probabilities on device: probs[i] = (1/dist[i]) / sum_j(1/dist[j])
__global__ void ga2opt_compute_fitness_prob_device(const GA2OptConfig cfg, const double* d_distances, double* d_probs) {
    extern __shared__ double sdata[]; // shared buffer for reduction

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (idx < cfg.n_population) {
        double d = d_distances[idx];
        // Guard against zero distance
        if (d <= 0.0) d = 1e-9;
        val = 1.0 / d;
        d_probs[idx] = val; // temporarily store raw fitness
    }

    // reduction within block
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Use first thread of each block to accumulate block sums.
    // NOTE: native double atomicAdd requires sm_60+. To stay portable, we
    // instead write block sums to a separate array and reduce on host.
    if (tid == 0) {
        d_probs[cfg.n_population + blockIdx.x] = sdata[0];
    }

    // After kernel, host will reduce d_probs[cfg.n_population .. cfg.n_population+gridDim.x-1]
    // to obtain total fitness, then call the normalize kernel.
}

// Normalize probabilities on device: probs[i] /= total
__global__ void ga2opt_normalize_probs_device(const GA2OptConfig cfg, double* d_probs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cfg.n_population) return;
    if (*cfg.d_total_fitness <= 0.0) *cfg.d_total_fitness = 1e-9;
    d_probs[idx] /= *cfg.d_total_fitness;
}

// Kernel for GPU-only 2-opt refinement of individuals.
// Each thread may or may not apply 2-opt to its chromosome depending
// on offspring_two_opt_prob.
__global__ void ga2opt_cuda_kernel_2opt(GA2OptConfig cfg, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cfg.n_population) return;

    unsigned int state = seed ^ (tid * 9781u + 1u);
    if (rand_real_device(state) < cfg.offspring_two_opt_prob) {
        int* route = cfg.d_population + tid * cfg.n_cities;
        two_opt_device(cfg, route, cfg.two_opt_passes_offspring);
        cfg.d_distances[tid] = d_total_distance(cfg, route, cfg.n_cities);
    }
}


__global__ void ga2opt_mutate_kernel(GA2OptConfig cfg, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cfg.n_population) return;
    if (tid == 0) return;  // 保留 elitism 的個體不 mutate （如果你要保留 index 0）

    unsigned int state = seed ^ (tid * 9781u + 1237u);
    int* chrom = cfg.d_next_population + tid * cfg.n_cities;

    // 單點 swap mutation（跟 host mutate_device 類似）
    if (rand_real_device(state) < cfg.mutation_rate) {
        int i = rand_int_device(state, 0, cfg.n_cities - 1);
        int j = rand_int_device(state, 0, cfg.n_cities - 1);
        int tmp = chrom[i];
        chrom[i] = chrom[j];
        chrom[j] = tmp;
    }
}

__global__ void ga2opt_eval_distance_kernel(GA2OptConfig cfg) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cfg.n_population) return;

    int* chrom = cfg.d_next_population + tid * cfg.n_cities;
    cfg.d_next_distances[tid] = d_total_distance(cfg, chrom, cfg.n_cities);
}

__global__ void ga2opt_select_and_crossover_kernel(
    GA2OptConfig cfg, const double* probs, unsigned int seed) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx1 = 1 + 2 * tid;
    int idx2 = idx1 + 1;
    if (idx1 >= cfg.n_population) return;

    unsigned int state = seed ^ (tid * 9781u + 1u);
    int p1_idx = roulette_device(cfg.n_population, probs, state);
    int p2_idx = roulette_device(cfg.n_population, probs, state);

    const int* p1 = cfg.d_population + p1_idx * cfg.n_cities;
    const int* p2 = cfg.d_population + p2_idx * cfg.n_cities;

    int* child1_ptr = cfg.d_next_population + idx1 * cfg.n_cities;
    int* child2_ptr = (idx2 < cfg.n_population) ? cfg.d_next_population + idx2 * cfg.n_cities : nullptr;

    if (rand_real_device(state) < cfg.crossover_rate) {
        crossover_device(cfg, state, p1, p2, child1_ptr, child2_ptr);
    } else {
        // no crossover: just copy parents
        for (int i = 0; i < cfg.n_cities; ++i) child1_ptr[i] = p1[i];
        if (child2_ptr) {
            const int* p2_only = cfg.d_population + p2_idx * cfg.n_cities;
            for (int i = 0; i < cfg.n_cities; ++i) child2_ptr[i] = p2_only[i];
        }
    }
    // 這裡「不做 mutation、不算距離」，讓 branch 減少
}

void host_fe_ga_2opt(int two_opt_passes_offspring,
                     double offspring_two_opt_prob) {
    if (offspring_two_opt_prob <= 0.0 || g_cfg.d_population == nullptr || g_cfg.d_distances == nullptr ||
        g_cfg.n_cities <= 0 || g_cfg.n_population <= 0) {
        return;
    }

    g_cfg.two_opt_passes_offspring = two_opt_passes_offspring;
    g_cfg.offspring_two_opt_prob = offspring_two_opt_prob;

    const int threads = 256;
    const int blocks = (g_cfg.n_population + threads - 1) / threads;

    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    ga2opt_cuda_kernel_2opt<<<blocks, threads>>>(g_cfg, seed);
    cudaDeviceSynchronize();
}

__global__ void ga2opt_elitism(GA2OptConfig g_cfg)
{
    // Device-side copy: cannot call cudaMemcpy from device code.
    int best = *g_cfg.d_best_idx;
    int* src = g_cfg.d_population + best * g_cfg.n_cities;
    int* dst = g_cfg.d_next_population; // copy into index 0 of next population
    for (int i = 0; i < g_cfg.n_cities; ++i) {
        dst[i] = src[i];
    }
    // copy distance for the elite into index 0
    g_cfg.d_next_distances[0] = g_cfg.d_distances[best];
}

// Full GA step on GPU: selection + crossover + mutation + distance evaluation.
// population/distances: current generation (size n_population * n_cities, n_population)
// next_population/next_distances: output generation; index 0 is elitism copy.
void host_fe_ga_selection_crossover_mutate(double crossover_rate,
                          double mutation_rate,
                          int two_opt_passes_offspring,
                          double offspring_two_opt_prob) {

    if (g_cfg.d_population == nullptr || g_cfg.d_distances == nullptr || g_cfg.d_cities == nullptr ||
        g_cfg.n_cities <= 0 || g_cfg.n_population <= 0) {
        return;
    }

    g_cfg.crossover_rate = crossover_rate;
    g_cfg.mutation_rate = mutation_rate;
    g_cfg.two_opt_passes_offspring = two_opt_passes_offspring;
    g_cfg.offspring_two_opt_prob = offspring_two_opt_prob;

    // 在 device 端找出當前世代的最佳個體，做真正的 elitism
    ga2opt_find_best_kernel<<<1, 1>>>(g_cfg, g_cfg.d_distances, g_cfg.d_best_idx);
    ga2opt_elitism<<<1, 1>>>(g_cfg);
    cudaDeviceSynchronize();

    const int threads = 256;
    const int blocks = (g_cfg.n_population + threads - 1) / threads;

    // 1) compute raw fitness and block partial sums on device
    ga2opt_compute_fitness_prob_device<<<blocks, threads, static_cast<size_t>(threads) * sizeof(double)>>>(g_cfg, g_cfg.d_distances, g_cfg.d_probs);
    cudaDeviceSynchronize();

    // 2) reduce partial sums on device (single thread kernel)
    ga2opt_reduce_total_fitness_kernel<<<1, 1>>>(g_cfg.d_probs + g_cfg.n_population, blocks, g_cfg.d_total_fitness);
    cudaDeviceSynchronize();

    // 3) normalize probabilities on device
    ga2opt_normalize_probs_device<<<blocks, threads>>>(g_cfg, g_cfg.d_probs);
    cudaDeviceSynchronize();

    // 4) selection + crossover into d_next_population (indices >=1)
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    ga2opt_select_and_crossover_kernel<<<blocks, threads>>>(g_cfg, g_cfg.d_probs, seed);
    cudaDeviceSynchronize();

    // 5) mutation on d_next_pop (except index 0)
    ga2opt_mutate_kernel<<<blocks, threads>>>(g_cfg, seed ^ 0x123456u);
    cudaDeviceSynchronize();

    // 6) evaluate distances for all next_population
    ga2opt_eval_distance_kernel<<<blocks, threads>>>(g_cfg);
    cudaDeviceSynchronize();

    // Swap current and next population buffers on device.
    int* tmp_pop = g_cfg.d_population;
    g_cfg.d_population = g_cfg.d_next_population;
    g_cfg.d_next_population = tmp_pop;

    double* tmp_dist = g_cfg.d_distances;
    g_cfg.d_distances = g_cfg.d_next_distances;
    g_cfg.d_next_distances = tmp_dist;
}

// New allocation API: allocate persistent device buffers for GA.
void host_fe_ga_allocate(const City* cities,
                         int n_cities,
                         int n_population)
{
    if (!cities || n_cities <= 0 || n_population <= 0) return;

    // If config already matches, only update city coordinates.
    if (g_cfg.d_cities && g_cfg.n_cities == n_cities && g_cfg.n_population == n_population) {
        const size_t cities_bytes = static_cast<size_t>(n_cities) * sizeof(City);
        cudaMemcpy(g_cfg.d_cities, cities, cities_bytes, cudaMemcpyHostToDevice);
        g_cfg.cities = cities;
        return;
    }

    // Otherwise free any existing buffers and reallocate.
    if (g_cfg.d_probs)          cudaFree(g_cfg.d_probs);
    if (g_cfg.d_next_distances) cudaFree(g_cfg.d_next_distances);
    if (g_cfg.d_next_population)cudaFree(g_cfg.d_next_population);
    if (g_cfg.d_distances)      cudaFree(g_cfg.d_distances);
    if (g_cfg.d_population)     cudaFree(g_cfg.d_population);
    if (g_cfg.d_dist_matrix)    cudaFree(g_cfg.d_dist_matrix);
    if (g_cfg.d_neighbors)      cudaFree(g_cfg.d_neighbors);
    if (g_cfg.d_cities)         cudaFree(g_cfg.d_cities);
    if (g_cfg.d_best_idx)       cudaFree(g_cfg.d_best_idx);
    if (g_cfg.d_total_fitness)  cudaFree(g_cfg.d_total_fitness);

    std::memset(&g_cfg, 0, sizeof(GA2OptConfig));
    g_cfg.cities = cities;
    g_cfg.n_cities = n_cities;
    g_cfg.n_population = n_population;

    const size_t cities_bytes = static_cast<size_t>(n_cities) * sizeof(City);
    const size_t dist_matrix_bytes = static_cast<size_t>(n_cities) * static_cast<size_t>(n_cities) * sizeof(double);
    const size_t pop_bytes = static_cast<size_t>(n_population) * static_cast<size_t>(n_cities) * sizeof(int);
    const size_t dist_bytes = static_cast<size_t>(n_population) * sizeof(double);

    cudaMalloc(&g_cfg.d_cities, cities_bytes);
    cudaMemcpy(g_cfg.d_cities, cities, cities_bytes, cudaMemcpyHostToDevice);

    // Build dense distance matrix on host and copy to device
    std::vector<double> h_dist_matrix(static_cast<size_t>(n_cities) * static_cast<size_t>(n_cities));
    for (int i = 0; i < n_cities; ++i) {
        h_dist_matrix[static_cast<size_t>(i) * static_cast<size_t>(n_cities) + static_cast<size_t>(i)] = 0.0;
        for (int j = i + 1; j < n_cities; ++j) {
            const double dx = cities[i].x - cities[j].x;
            const double dy = cities[i].y - cities[j].y;
            const double d = std::sqrt(dx * dx + dy * dy);
            const size_t idx1 = static_cast<size_t>(i) * static_cast<size_t>(n_cities) + static_cast<size_t>(j);
            const size_t idx2 = static_cast<size_t>(j) * static_cast<size_t>(n_cities) + static_cast<size_t>(i);
            h_dist_matrix[idx1] = d;
            h_dist_matrix[idx2] = d;
        }
    }

    cudaMalloc(&g_cfg.d_dist_matrix, dist_matrix_bytes);
    cudaMemcpy(g_cfg.d_dist_matrix, h_dist_matrix.data(), dist_matrix_bytes, cudaMemcpyHostToDevice);

    // Build nearest-neighbor candidate list on host and copy to device.
    // Use up to M neighbors per city (exclude self).
    int M = std::min(8, n_cities - 1);
    if (M < 0) M = 0;
    g_cfg.n_neighbors = M;
    if (M > 0) {
        std::vector<int> h_neighbors(static_cast<size_t>(n_cities) * static_cast<size_t>(M));
        for (int i = 0; i < n_cities; ++i) {
            std::vector<std::pair<double,int>> cand;
            cand.reserve(n_cities - 1);
            for (int j = 0; j < n_cities; ++j) {
                if (j == i) continue;
                cand.emplace_back(h_dist_matrix[static_cast<size_t>(i) * static_cast<size_t>(n_cities) + static_cast<size_t>(j)], j);
            }
            if ((int)cand.size() > M) {
                std::partial_sort(cand.begin(), cand.begin() + M, cand.end(), [](const std::pair<double,int>& a, const std::pair<double,int>& b){ return a.first < b.first; });
            } else {
                std::sort(cand.begin(), cand.end(), [](const std::pair<double,int>& a, const std::pair<double,int>& b){ return a.first < b.first; });
            }
            for (int m = 0; m < M; ++m) {
                h_neighbors[static_cast<size_t>(i) * static_cast<size_t>(M) + static_cast<size_t>(m)] = cand[m].second;
            }
        }
        cudaMalloc(&g_cfg.d_neighbors, static_cast<size_t>(n_cities) * static_cast<size_t>(M) * sizeof(int));
        cudaMemcpy(g_cfg.d_neighbors, h_neighbors.data(), static_cast<size_t>(n_cities) * static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);
    } else {
        g_cfg.d_neighbors = nullptr;
        g_cfg.n_neighbors = 0;
    }

    cudaMalloc(&g_cfg.d_population, pop_bytes);
    cudaMalloc(&g_cfg.d_next_population, pop_bytes);
    cudaMalloc(&g_cfg.d_distances, dist_bytes);
    cudaMalloc(&g_cfg.d_next_distances, dist_bytes);
    cudaMalloc(&g_cfg.d_best_idx, sizeof(int));

    const int threads = 256;
    const int blocks = (n_population + threads - 1) / threads;
    cudaMalloc(&g_cfg.d_probs, static_cast<size_t>(n_population + blocks) * sizeof(double));
    cudaMalloc(&g_cfg.d_total_fitness, sizeof(double));
}

void host_fe_ga_free()
{
    if (g_cfg.d_probs)          cudaFree(g_cfg.d_probs);
    if (g_cfg.d_next_distances) cudaFree(g_cfg.d_next_distances);
    if (g_cfg.d_next_population)cudaFree(g_cfg.d_next_population);
    if (g_cfg.d_distances)      cudaFree(g_cfg.d_distances);
    if (g_cfg.d_population)     cudaFree(g_cfg.d_population);
    if (g_cfg.d_dist_matrix)    cudaFree(g_cfg.d_dist_matrix);
    if (g_cfg.d_neighbors)      cudaFree(g_cfg.d_neighbors);
    if (g_cfg.d_cities)         cudaFree(g_cfg.d_cities);
    if (g_cfg.d_best_idx)       cudaFree(g_cfg.d_best_idx);
    if (g_cfg.d_total_fitness)  cudaFree(g_cfg.d_total_fitness);

    std::memset(&g_cfg, 0, sizeof(GA2OptConfig));
}

void host_fe_ga_copy_population_to_device(const int* population,
                                          const double* distances)
{
    if (!population || !distances || g_cfg.d_population == nullptr || g_cfg.d_distances == nullptr ||
        g_cfg.n_cities <= 0 || g_cfg.n_population <= 0) {
        return;
    }

    const size_t pop_bytes = static_cast<size_t>(g_cfg.n_population) * static_cast<size_t>(g_cfg.n_cities) * sizeof(int);
    const size_t dist_bytes = static_cast<size_t>(g_cfg.n_population) * sizeof(double);

    cudaMemcpy(g_cfg.d_population, population, pop_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_cfg.d_distances, distances, dist_bytes, cudaMemcpyHostToDevice);
}

void host_fe_ga_copy_population_to_host(int* population,
                                        double* distances)
{
    if (!population || !distances || g_cfg.d_population == nullptr || g_cfg.d_distances == nullptr ||
        g_cfg.n_cities <= 0 || g_cfg.n_population <= 0) {
        return;
    }

    const size_t pop_bytes = static_cast<size_t>(g_cfg.n_population) * static_cast<size_t>(g_cfg.n_cities) * sizeof(int);
    const size_t dist_bytes = static_cast<size_t>(g_cfg.n_population) * sizeof(double);

    cudaMemcpy(population, g_cfg.d_population, pop_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, g_cfg.d_distances, dist_bytes, cudaMemcpyDeviceToHost);
}