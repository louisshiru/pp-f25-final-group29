#include "dataloader.h"
#include "kernel.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ctime>

#define GEN_COLLECT_INTERVAL 100

// ======================== Cuda implementation ========================

__device__ double d_dist_city(const City& a, const City& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

__device__ double d_total_distance(const int* route, int n, const City* cities) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const City& c1 = cities[route[i]];
        const City& c2 = cities[route[(i + 1) % n]];
        total += d_dist_city(c1, c2);
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

__device__ void two_opt_device(const GA2OptConfig& cfg, int* route, int max_passes, const City* d_cities) {

    if (cfg.n_cities < 4) return;
    bool improved = true;
    int passes = 0;
    while (improved && passes < max_passes) {
        improved = false;
        for (int i = 1; i < cfg.n_cities - 1 && !improved; ++i) {
            for (int k = i + 1; k < cfg.n_cities && !improved; ++k) {
                int next = (k + 1) % cfg.n_cities;
                double delta =
                    d_dist_city(d_cities[route[i - 1]], d_cities[route[k]]) +
                    d_dist_city(d_cities[route[i]], d_cities[route[next]]) -
                    d_dist_city(d_cities[route[i - 1]], d_cities[route[i]]) -
                    d_dist_city(d_cities[route[k]], d_cities[route[next]]);
                if (delta < -1e-9) {
                    d_reverse_segment(route, i, k);
                    improved = true;
                }
            }
        }
        ++passes;
    }
};

__device__ void crossover_device(const GA2OptConfig& cfg, unsigned int& state, const int* p1, const int* p2, int* c1, int* c2) {
    int cut1 = rand_int_device(state, 0, cfg.n_cities - 1);
    int cut2 = rand_int_device(state, 0, cfg.n_cities - 1);
    if (cut1 > cut2) { int t = cut1; cut1 = cut2; cut2 = t; }
    const int MAX_CITIES = 4096;  // adjust if you ever need larger instances
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

__device__ void fitness_device(const GA2OptConfig& cfg, const int* chrom, double* out_distance, const City* d_cities) {
    *out_distance = d_total_distance(chrom, cfg.n_cities, d_cities);
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
__global__ void ga2opt_normalize_probs_device(const GA2OptConfig cfg, double* d_probs, double total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cfg.n_population) return;
    if (total <= 0.0) total = 1e-9;
    d_probs[idx] /= total;
}

// NOTE: this kernel is a straightforward, not fully optimized port of ga2opt_run.
// One thread works on one or two offspring based on its thread id.
__global__ void ga2opt_cuda_kernel(GA2OptConfig cfg, const double* probs, const City* d_cities, unsigned int seed) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int idx1 = 1 + 2 * tid;
    int idx2 = idx1 + 1;
    if (idx1 >= cfg.n_population) return;

    // Very simple xorshift RNG per thread
    unsigned int state = seed ^ (tid * 9781u + 1u);
    int p1_idx = roulette_device(cfg.n_population, probs, state);
    int p2_idx = roulette_device(cfg.n_population, probs, state);

    int* child1_ptr = cfg.d_next_population + idx1 * cfg.n_cities;
    int* child2_ptr = (idx2 < cfg.n_population) ? cfg.d_next_population + idx2 * cfg.n_cities : nullptr;

    if (rand_real_device(state) < cfg.crossover_rate) {
        const int* p1 = cfg.d_population + p1_idx * cfg.n_cities;
        const int* p2 = cfg.d_population + p2_idx * cfg.n_cities;
        crossover_device(cfg, state, p1, p2, child1_ptr, child2_ptr);

        mutate_device(cfg, child1_ptr, state);
        two_opt_device(cfg, child1_ptr, cfg.two_opt_passes_offspring, d_cities);

        cfg.d_next_distances[idx1] = d_total_distance(child1_ptr, cfg.n_cities, d_cities);

        if (child2_ptr) {
            mutate_device(cfg, child2_ptr, state);
            two_opt_device(cfg, child2_ptr, cfg.two_opt_passes_offspring, d_cities);
            cfg.d_next_distances[idx2] = d_total_distance(child2_ptr, cfg.n_cities, d_cities);
        }
    } else {
        const int* p1 = cfg.d_population + p1_idx * cfg.n_cities;
        for (int i = 0; i < cfg.n_cities; ++i) child1_ptr[i] = p1[i];
        mutate_device(cfg, child1_ptr, state);
        two_opt_device(cfg, child1_ptr, cfg.two_opt_passes_offspring, d_cities);
        cfg.d_next_distances[idx1] = d_total_distance(child1_ptr, cfg.n_cities, d_cities);

        if (child2_ptr) {
            const int* p2 = cfg.d_population + p2_idx * cfg.n_cities;
            for (int i = 0; i < cfg.n_cities; ++i) child2_ptr[i] = p2[i];
            mutate_device(cfg, child2_ptr, state);
            two_opt_device(cfg, child2_ptr, cfg.two_opt_passes_offspring, d_cities);
            cfg.d_next_distances[idx2] = d_total_distance(child2_ptr, cfg.n_cities, d_cities);
        }
    }
}

// Kernel for GPU-only 2-opt refinement of individuals.
// Each thread may or may not apply 2-opt to its chromosome depending
// on offspring_two_opt_prob.
__global__ void ga2opt_cuda_kernel_2opt_only(GA2OptConfig cfg, const City* d_cities, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cfg.n_population) return;

    unsigned int state = seed ^ (tid * 9781u + 1u);
    if (rand_real_device(state) < cfg.offspring_two_opt_prob) {
        int* route = cfg.d_population + tid * cfg.n_cities;
        two_opt_device(cfg, route, cfg.two_opt_passes_offspring, d_cities);
        cfg.d_distances[tid] = d_total_distance(route, cfg.n_cities, d_cities);
    }
}

__global__ void ga2opt_cuda_kernel_no_2opt(GA2OptConfig cfg, const double* probs, const City* d_cities, unsigned int seed) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int idx1 = 1 + 2 * tid;
    int idx2 = idx1 + 1;
    if (idx1 >= cfg.n_population) return;

    // Very simple xorshift RNG per thread
    unsigned int state = seed ^ (tid * 9781u + 1u);
    int p1_idx = roulette_device(cfg.n_population, probs, state);
    int p2_idx = roulette_device(cfg.n_population, probs, state);

    int* child1_ptr = cfg.d_next_population + idx1 * cfg.n_cities;
    int* child2_ptr = (idx2 < cfg.n_population) ? cfg.d_next_population + idx2 * cfg.n_cities : nullptr;

    if (rand_real_device(state) < cfg.crossover_rate) {
        const int* p1 = cfg.d_population + p1_idx * cfg.n_cities;
        const int* p2 = cfg.d_population + p2_idx * cfg.n_cities;
        crossover_device(cfg, state, p1, p2, child1_ptr, child2_ptr);
        mutate_device(cfg, child1_ptr, state);

        cfg.d_next_distances[idx1] = d_total_distance(child1_ptr, cfg.n_cities, d_cities);

        if (child2_ptr) {
            mutate_device(cfg, child2_ptr, state);
            cfg.d_next_distances[idx2] = d_total_distance(child2_ptr, cfg.n_cities, d_cities);
        }
    } else {
        const int* p1 = cfg.d_population + p1_idx * cfg.n_cities;
        for (int i = 0; i < cfg.n_cities; ++i) child1_ptr[i] = p1[i];
        mutate_device(cfg, child1_ptr, state);
        cfg.d_next_distances[idx1] = d_total_distance(child1_ptr, cfg.n_cities, d_cities);

        if (child2_ptr) {
            const int* p2 = cfg.d_population + p2_idx * cfg.n_cities;
            for (int i = 0; i < cfg.n_cities; ++i) child2_ptr[i] = p2[i];
            mutate_device(cfg, child2_ptr, state);
            cfg.d_next_distances[idx2] = d_total_distance(child2_ptr, cfg.n_cities, d_cities);
        }
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

__global__ void ga2opt_eval_distance_kernel(GA2OptConfig cfg, const City* d_cities) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cfg.n_population) return;

    int* chrom = cfg.d_next_population + tid * cfg.n_cities;
    cfg.d_next_distances[tid] = d_total_distance(chrom, cfg.n_cities, d_cities);
}

__global__ void ga2opt_select_and_crossover_kernel(
    GA2OptConfig cfg, const double* probs, const City* d_cities, unsigned int seed) {

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

void host_fe_ga_2opt(const City* cities,
                     int n_cities,
                     int n_population,
                     int two_opt_passes_offspring,
                     double offspring_two_opt_prob,
                     int* population,
                     double* distances) {
    if (!cities || !population || !distances || n_cities <= 0 || n_population <= 0) {
        return;
    }

    GA2OptConfig cfg;
    cfg.cities = cities;
    cfg.n_cities = n_cities;
    cfg.n_population = n_population;
    cfg.n_generations = 0;
    cfg.crossover_rate = 0.0;
    cfg.mutation_rate = 0.0;
    cfg.init_two_opt_prob = 0.0;
    cfg.offspring_two_opt_prob = 0.0;
    cfg.two_opt_passes_init = 0;
    cfg.two_opt_passes_offspring = two_opt_passes_offspring;
    cfg.population = population;
    cfg.distances = distances;
    cfg.next_population = nullptr;
    cfg.next_distances = nullptr;
    cfg.d_population = nullptr;
    cfg.d_next_population = nullptr;
    cfg.d_distances = nullptr;
    cfg.d_next_distances = nullptr;

    const size_t cities_bytes = static_cast<size_t>(n_cities) * sizeof(City);
    const size_t pop_bytes = static_cast<size_t>(n_population) * static_cast<size_t>(n_cities) * sizeof(int);
    const size_t dist_bytes = static_cast<size_t>(n_population) * sizeof(double);

    City* d_cities = nullptr;
    cudaMalloc(&d_cities, cities_bytes);
    cudaMemcpy(d_cities, cities, cities_bytes, cudaMemcpyHostToDevice);

    int* d_population = nullptr;
    double* d_distances = nullptr;
    cudaMalloc(&d_population, pop_bytes);
    cudaMalloc(&d_distances, dist_bytes);
    cudaMemcpy(d_population, population, pop_bytes, cudaMemcpyHostToDevice);

    cfg.d_population = d_population;
    cfg.d_distances = d_distances;
    cfg.offspring_two_opt_prob = offspring_two_opt_prob;

    const int threads = 256;
    const int blocks = (n_population + threads - 1) / threads;

    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    ga2opt_cuda_kernel_2opt_only<<<blocks, threads>>>(cfg, d_cities, seed);
    cudaDeviceSynchronize();

    cudaMemcpy(population, d_population, pop_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, dist_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_cities);
    cudaFree(d_population);
    cudaFree(d_distances);
}