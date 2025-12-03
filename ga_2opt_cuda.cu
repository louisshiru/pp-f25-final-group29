#include "dataloader.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ctime>

// Simple 2-opt local search used as a GA refinement step.
// Given a permutation of city indices, it keeps swapping two edges
// whenever the total tour length can be shortened.
double dist_city(const City& a, const City& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

double total_distance(const int* route, int n, const City* cities) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const City& c1 = cities[route[i]];
        const City& c2 = cities[route[(i + 1) % n]];
        total += dist_city(c1, c2);
    }
    return total;
}

void reverse_segment(int* route, int i, int k) {
    while (i < k) {
        int temp = route[i];
        route[i] = route[k];
        route[k] = temp;
        i++;
        k--;
    }
}

// A capped 2-opt: at most `max_passes` improvement rounds to avoid long runtimes on large instances.
void two_opt(int* route, int n, const City* cities, int max_passes) {
    if (n < 4) return;

    bool improved = true;
    int passes = 0;
    while (improved && passes < max_passes) {
        improved = false;
        for (int i = 1; i < n - 1 && !improved; ++i) {
            for (int k = i + 1; k < n && !improved; ++k) {
                const int next = (k + 1) % n;
                const double delta =
                    dist_city(cities[route[i - 1]], cities[route[k]]) +
                    dist_city(cities[route[i]], cities[route[next]]) -
                    dist_city(cities[route[i - 1]], cities[route[i]]) -
                    dist_city(cities[route[k]], cities[route[next]]);

                if (delta < -1e-9) {
                    reverse_segment(route, i, k);
                    improved = true;
                }
            }
        }
        ++passes;
    }
}

struct GA2OptConfig {
    const City* cities;
    int n_cities;
    int n_population;
    int n_generations;
    double crossover_rate;
    double mutation_rate;
    double init_two_opt_prob;
    double offspring_two_opt_prob;
    int two_opt_passes_init;
    int two_opt_passes_offspring;

    int* population;
    double* distances;
    int* next_population;
    double* next_distances;

    // Device copies (for CUDA version)
    int* d_population = nullptr;
    int* d_next_population = nullptr;
    double* d_distances = nullptr;
    double* d_next_distances = nullptr;
};

double ga2opt_random_real() {
    return (double)rand() / RAND_MAX;
}

int ga2opt_random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

void ga2opt_shuffle_array(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = ga2opt_random_int(0, i);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void ga2opt_initial_population(GA2OptConfig& cfg) {
    int* base = new int[cfg.n_cities];
    for (int i = 0; i < cfg.n_cities; ++i) base[i] = i;

    for (int i = 0; i < cfg.n_population; ++i) {
        int* chrom = cfg.population + i * cfg.n_cities;
        for (int j = 0; j < cfg.n_cities; ++j) chrom[j] = base[j];

        ga2opt_shuffle_array(chrom, cfg.n_cities);

        if (ga2opt_random_real() < cfg.init_two_opt_prob) {
            two_opt(chrom, cfg.n_cities, cfg.cities, cfg.two_opt_passes_init);
        }
        cfg.distances[i] = total_distance(chrom, cfg.n_cities, cfg.cities);
    }
    delete[] base;
}

void ga2opt_compute_fitness_prob(const GA2OptConfig& cfg, double* probs) {
    double total_fitness = 0.0;
    for (int i = 0; i < cfg.n_population; ++i) {
        double f = 1.0 / cfg.distances[i];
        probs[i] = f;
        total_fitness += f;
    }
    for (int i = 0; i < cfg.n_population; ++i) {
        probs[i] /= total_fitness;
    }
}

int ga2opt_roulette_wheel(const GA2OptConfig& cfg, const double* probs) {
    double r = ga2opt_random_real();
    double cum_prob = 0.0;
    for (int i = 0; i < cfg.n_population; ++i) {
        cum_prob += probs[i];
        if (r <= cum_prob) {
            return i;
        }
    }
    return cfg.n_population - 1;
}

int ga2opt_find_best_individual(const GA2OptConfig& cfg) {
    int best_idx = 0;
    for (int i = 1; i < cfg.n_population; ++i) {
        // printf("Individual %d: Distance = %f\n", i, cfg.distances[i]);
        if (cfg.distances[i] < cfg.distances[best_idx]) {
            best_idx = i;
        }
    }
    return best_idx;
}

void ga2opt_copy_individual(const GA2OptConfig& cfg, const int* src, int* dest) {
    for (int i = 0; i < cfg.n_cities; ++i) {
        dest[i] = src[i];
    }
}

void ga2opt_create_child_helper(const GA2OptConfig& cfg, const int* parent_a, const int* parent_b,
                                int* child, int cut1, int cut2, bool* used) {
    for (int i = 0; i < cfg.n_cities; ++i) {
        child[i] = -1;
        used[i] = false;
    }

    for (int i = cut1; i <= cut2; ++i) {
        child[i] = parent_a[i];
        used[parent_a[i]] = true;
    }

    int idx_b = (cut2 + 1) % cfg.n_cities;
    int idx_child = (cut2 + 1) % cfg.n_cities;
    for (int i = 0; i < cfg.n_cities; ++i) {
        int candidate = parent_b[idx_b];
        if (!used[candidate]) {
            child[idx_child] = candidate;
            used[candidate] = true;
            idx_child = (idx_child + 1) % cfg.n_cities;
        }
        idx_b = (idx_b + 1) % cfg.n_cities;
    }
}

void ga2opt_crossover(const GA2OptConfig& cfg, const int* p1, const int* p2, int* c1, int* c2) {
    int cut1 = ga2opt_random_int(0, cfg.n_cities - 1);
    int cut2 = ga2opt_random_int(0, cfg.n_cities - 1);
    if (cut1 > cut2) {
        int t = cut1; cut1 = cut2; cut2 = t;
    }

    bool* used = new bool[cfg.n_cities];

    ga2opt_create_child_helper(cfg, p1, p2, c1, cut1, cut2, used);
    if (c2) {
        ga2opt_create_child_helper(cfg, p2, p1, c2, cut1, cut2, used);
    }

    delete[] used;
}

void ga2opt_mutate(const GA2OptConfig& cfg, int* chrom) {
    if (ga2opt_random_real() < cfg.mutation_rate) {
        int i = ga2opt_random_int(0, cfg.n_cities - 1);
        int j = ga2opt_random_int(0, cfg.n_cities - 1);
        int temp = chrom[i];
        chrom[i] = chrom[j];
        chrom[j] = temp;
    }
}


void ga2opt_run(GA2OptConfig& cfg) {
    double* probs = new double[cfg.n_population];
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < cfg.n_generations; ++gen) {
        ga2opt_compute_fitness_prob(cfg, probs);

        // Elitism: keep the current best individual at index 0 of next_population_.
        int best_idx = ga2opt_find_best_individual(cfg);
        ga2opt_copy_individual(cfg, cfg.population + best_idx * cfg.n_cities,
                               cfg.next_population + 0 * cfg.n_cities);
        cfg.next_distances[0] = cfg.distances[best_idx];

        // Remaining positions [1, n_population_ - 1] are filled by offspring.
        // We conceptually have "units" where each unit can generate up to 2 children.
        int remaining = cfg.n_population - 1;
        int n_units = (remaining + 1) / 2; // ceil(remaining / 2)

        for (int u = 0; u < n_units; ++u) {
            int idx1 = 1 + 2 * u;
            int idx2 = idx1 + 1;
            if (idx1 >= cfg.n_population) break;

            int p1_idx = ga2opt_roulette_wheel(cfg, probs);
            int p2_idx = ga2opt_roulette_wheel(cfg, probs);

            int* child1_ptr = cfg.next_population + idx1 * cfg.n_cities;
            int* child2_ptr = (idx2 < cfg.n_population) ? cfg.next_population + idx2 * cfg.n_cities : nullptr;

            if (ga2opt_random_real() < cfg.crossover_rate) {
                ga2opt_crossover(cfg,
                                 cfg.population + p1_idx * cfg.n_cities,
                                 cfg.population + p2_idx * cfg.n_cities,
                                 child1_ptr, child2_ptr);

                ga2opt_mutate(cfg, child1_ptr);
                two_opt(child1_ptr, cfg.n_cities, cfg.cities, cfg.two_opt_passes_offspring);
                cfg.next_distances[idx1] = total_distance(child1_ptr, cfg.n_cities, cfg.cities);

                if (child2_ptr) {
                    ga2opt_mutate(cfg, child2_ptr);
                    two_opt(child2_ptr, cfg.n_cities, cfg.cities, cfg.two_opt_passes_offspring);
                    cfg.next_distances[idx2] = total_distance(child2_ptr, cfg.n_cities, cfg.cities);
                }
            } else {
                ga2opt_copy_individual(cfg, cfg.population + p1_idx * cfg.n_cities, child1_ptr);
                ga2opt_mutate(cfg, child1_ptr);
                two_opt(child1_ptr, cfg.n_cities, cfg.cities, cfg.two_opt_passes_offspring);
                cfg.next_distances[idx1] = total_distance(child1_ptr, cfg.n_cities, cfg.cities);

                if (child2_ptr) {
                    ga2opt_copy_individual(cfg, cfg.population + p2_idx * cfg.n_cities, child2_ptr);
                    ga2opt_mutate(cfg, child2_ptr);
                    two_opt(child2_ptr, cfg.n_cities, cfg.cities, cfg.two_opt_passes_offspring);
                    cfg.next_distances[idx2] = total_distance(child2_ptr, cfg.n_cities, cfg.cities);
                }
            }
        }

        // Swap populations
        int* temp_pop = cfg.population;
        cfg.population = cfg.next_population;
        cfg.next_population = temp_pop;

        double* temp_dist = cfg.distances;
        cfg.distances = cfg.next_distances;
        cfg.next_distances = temp_dist;

        int current_best_idx = ga2opt_find_best_individual(cfg);
        double best_dist = cfg.distances[current_best_idx];

        if (gen % 100 == 0 || gen == cfg.n_generations - 1) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            start_time = current_time;
            std::cout << "Generation " << gen << " Best Distance: " << best_dist << " Time: " << elapsed.count() << "s" << std::endl;
        }
    }

    int final_best_idx = ga2opt_find_best_individual(cfg);
    std::cout << "Final Best Distance: " << cfg.distances[final_best_idx] << std::endl;
    std::cout << "Best Route: ";
    int* best_chrom = cfg.population + final_best_idx * cfg.n_cities;
    for (int i = 0; i < cfg.n_cities; ++i) {
        std::cout << best_chrom[i] << " ";
    }
    std::cout << best_chrom[0] << std::endl; // close the tour
    
    delete[] probs;
}


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

void ga2opt_run_cuda(GA2OptConfig& cfg) {
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaError_t err = cudaGetLastError();
    std::cout << "Running on GPU: " << prop.name << std::endl;
    std::cout << "Max threads block: " << prop.maxThreadsPerBlock << std::endl;

    double* probs = new double[cfg.n_population];

    // Allocate device memory for populations and distances
    size_t pop_bytes = (size_t)cfg.n_population * (size_t)cfg.n_cities * sizeof(int);
    size_t dist_bytes = (size_t)cfg.n_population * sizeof(double);

    cudaMalloc(&cfg.d_population, pop_bytes);
    cudaMalloc(&cfg.d_next_population, pop_bytes);
    cudaMalloc(&cfg.d_distances, dist_bytes);
    cudaMalloc(&cfg.d_next_distances, dist_bytes);

    // Copy initial population and distances to device
    cudaMemcpy(cfg.d_population, cfg.population, pop_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cfg.d_distances, cfg.distances, dist_bytes, cudaMemcpyHostToDevice);

    // Copy cities to device (constant across generations)
    City* d_cities = nullptr;
    cudaMalloc(&d_cities, (size_t)cfg.n_cities * sizeof(City));
    cudaMemcpy(d_cities, cfg.cities, (size_t)cfg.n_cities * sizeof(City), cudaMemcpyHostToDevice);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < cfg.n_generations; ++gen) {
        ga2opt_compute_fitness_prob(cfg, probs);

        // Copy fitness probabilities to device
        double* d_probs = nullptr;
        cudaMalloc(&d_probs, dist_bytes);
        cudaMemcpy(d_probs, probs, dist_bytes, cudaMemcpyHostToDevice);
 
        // Elitism on host: copy best individual to next_population[0]
        int best_idx = ga2opt_find_best_individual(cfg);
        ga2opt_copy_individual(cfg, cfg.population + best_idx * cfg.n_cities,
                       cfg.next_population + 0 * cfg.n_cities);
        cfg.next_distances[0] = cfg.distances[best_idx];

        // Copy elite to device next population
        cudaMemcpy(cfg.d_next_population, cfg.next_population, (size_t)cfg.n_cities * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cfg.d_next_distances, cfg.next_distances, sizeof(double), cudaMemcpyHostToDevice);

        unsigned int seed = (unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        int blockSizeOffspring = 256;
        int n_units = (cfg.n_population - 1 + 1) / 2; // ceil((n_population-1)/2)
        int gridSizeOffspring = (n_units + blockSizeOffspring - 1) / blockSizeOffspring;
        ga2opt_cuda_kernel<<<gridSizeOffspring, blockSizeOffspring>>>(cfg, d_probs, d_cities, seed);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Memory failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaFree(d_probs);

        // Copy updated populations and distances back to host
        cudaMemcpy(cfg.next_population, cfg.d_next_population, pop_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(cfg.next_distances, cfg.d_next_distances, dist_bytes, cudaMemcpyDeviceToHost);

        // Swap populations (host side; device pointers remain mapped to logical roles)
        int* temp_pop = cfg.population;
        cfg.population = cfg.next_population;
        cfg.next_population = temp_pop;

        double* temp_dist = cfg.distances;
        cfg.distances = cfg.next_distances;
        cfg.next_distances = temp_dist;

        int current_best_idx = ga2opt_find_best_individual(cfg);
        double best_dist = cfg.distances[current_best_idx];
        if (gen % 100 == 0 || gen == cfg.n_generations - 1) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            start_time = current_time;
            std::cout << "Generation " << gen << " Best Distance: " << best_dist << " Time: " << elapsed.count() << "s" << std::endl;
        }
    }

    int final_best_idx = ga2opt_find_best_individual(cfg);
    std::cout << "Final Best Distance: " << cfg.distances[final_best_idx] << std::endl;
    std::cout << "Best Route: ";
    int* best_chrom = cfg.population + final_best_idx * cfg.n_cities;
    for (int i = 0; i < cfg.n_cities; ++i) {
        std::cout << best_chrom[i] << " ";
    }
    std::cout << best_chrom[0] << std::endl; // close the tour
    
    delete[] probs;
}

// Kernel for GPU-only 2-opt refinement of all individuals.
// Each thread takes one chromosome from d_population, applies 2-opt, and
// writes the improved tour and its distance to d_next_population/d_next_distances.
__global__ void ga2opt_cuda_kernel_2opt_only(GA2OptConfig cfg, const City* d_cities) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cfg.n_population) return;

    int* route = cfg.d_population + tid * cfg.n_cities;
    two_opt_device(cfg, route, cfg.two_opt_passes_offspring, d_cities);
    cfg.d_distances[tid] = d_total_distance(route, cfg.n_cities, d_cities);
}

// Version 2: GA on CPU, 2-opt local search on GPU only.
void ga2opt_run_cuda2(GA2OptConfig& cfg) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaError_t err = cudaGetLastError();
    std::cout << "Running GA (CPU) + 2-opt (GPU) on: " << prop.name << std::endl;

    // Allocate device-side storage for population, distances, and cities
    size_t pop_bytes = (size_t)cfg.n_population * (size_t)cfg.n_cities * sizeof(int);
    size_t dist_bytes = (size_t)cfg.n_population * sizeof(double);

    cudaMalloc(&cfg.d_population, pop_bytes);
    cudaMalloc(&cfg.d_distances, dist_bytes);

    City* d_cities = nullptr;
    cudaMalloc(&d_cities, (size_t)cfg.n_cities * sizeof(City));
    cudaMemcpy(d_cities, cfg.cities, (size_t)cfg.n_cities * sizeof(City), cudaMemcpyHostToDevice);

    auto start_time = std::chrono::high_resolution_clock::now();

    double* probs = new double[cfg.n_population];

    for (int gen = 0; gen < cfg.n_generations; ++gen) {
        // GA selection / crossover / mutation fully on CPU, but WITHOUT 2-opt.
        ga2opt_compute_fitness_prob(cfg, probs);

        // Elitism on CPU: copy best individual to next_population[0]
        int best_idx = ga2opt_find_best_individual(cfg);
        ga2opt_copy_individual(cfg, cfg.population + best_idx * cfg.n_cities,
                               cfg.next_population + 0 * cfg.n_cities);
        cfg.next_distances[0] = cfg.distances[best_idx];

        int remaining = cfg.n_population - 1;
        int n_units = (remaining + 1) / 2;

        for (int u = 0; u < n_units; ++u) {
            int idx1 = 1 + 2 * u;
            int idx2 = idx1 + 1;
            if (idx1 >= cfg.n_population) break;

            int p1_idx = ga2opt_roulette_wheel(cfg, probs);
            int p2_idx = ga2opt_roulette_wheel(cfg, probs);

            int* child1_ptr = cfg.next_population + idx1 * cfg.n_cities;
            int* child2_ptr = (idx2 < cfg.n_population) ? cfg.next_population + idx2 * cfg.n_cities : nullptr;

            if (ga2opt_random_real() < cfg.crossover_rate) {
                ga2opt_crossover(cfg,
                                 cfg.population + p1_idx * cfg.n_cities,
                                 cfg.population + p2_idx * cfg.n_cities,
                                 child1_ptr, child2_ptr);

                ga2opt_mutate(cfg, child1_ptr);
                if (child2_ptr) {
                    ga2opt_mutate(cfg, child2_ptr);
                }
            } else {
                ga2opt_copy_individual(cfg, cfg.population + p1_idx * cfg.n_cities, child1_ptr);
                ga2opt_mutate(cfg, child1_ptr);
                if (child2_ptr) {
                    ga2opt_copy_individual(cfg, cfg.population + p2_idx * cfg.n_cities, child2_ptr);
                    ga2opt_mutate(cfg, child2_ptr);
                }
            }
        }

        // Now offload ONLY 2-opt to GPU for every individual in next_population
        cudaMemcpy(cfg.d_population, cfg.next_population, pop_bytes, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (cfg.n_population + blockSize - 1) / blockSize;
        ga2opt_cuda_kernel_2opt_only<<<gridSize, blockSize>>>(cfg, d_cities);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel2 launch failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel2 sync failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        // Bring improved population + distances back to host
        cudaMemcpy(cfg.next_population, cfg.d_population, pop_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(cfg.next_distances, cfg.d_distances, dist_bytes, cudaMemcpyDeviceToHost);

        // Swap populations on host
        int* temp_pop = cfg.population;
        cfg.population = cfg.next_population;
        cfg.next_population = temp_pop;

        double* temp_dist = cfg.distances;
        cfg.distances = cfg.next_distances;
        cfg.next_distances = temp_dist;

        int current_best_idx = ga2opt_find_best_individual(cfg);
        double best_dist = cfg.distances[current_best_idx];
        if (gen % 100 == 0 || gen == cfg.n_generations - 1) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            start_time = current_time;
            std::cout << "[CPU-GA / GPU-2opt] Generation " << gen
                      << " Best Distance: " << best_dist
                      << " Time: " << elapsed.count() << "s" << std::endl;
        }
    }

    int final_best_idx = ga2opt_find_best_individual(cfg);
    std::cout << "[CPU-GA / GPU-2opt] Final Best Distance: " << cfg.distances[final_best_idx] << std::endl;
    std::cout << "Best Route: ";
    int* best_chrom = cfg.population + final_best_idx * cfg.n_cities;
    for (int i = 0; i < cfg.n_cities; ++i) {
        std::cout << best_chrom[i] << " ";
    }
    std::cout << best_chrom[0] << std::endl;

    // Cleanup
    delete[] probs;
    cudaFree(cfg.d_population);
    cudaFree(cfg.d_distances);
    cudaFree(d_cities);
}

// Version3 : GA, 2-opt both on GPU and Collect results every 100 generations
void ga2opt_run_cuda3(GA2OptConfig& cfg) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaError_t err = cudaGetLastError();
    std::cout << "[GPU-GA+2opt] Running on GPU: " << prop.name << std::endl;

    // Allocate device memory for populations and distances (if not already allocated)
    size_t pop_bytes = (size_t)cfg.n_population * (size_t)cfg.n_cities * sizeof(int);
    size_t dist_bytes = (size_t)cfg.n_population * sizeof(double);
    // extra gridSize slots for per-block partial sums (computed later)

    cudaMalloc(&cfg.d_population, pop_bytes);
    cudaMalloc(&cfg.d_next_population, pop_bytes);
    cudaMalloc(&cfg.d_distances, dist_bytes);
    cudaMalloc(&cfg.d_next_distances, dist_bytes);

    // Copy initial population and distances to device
    cudaMemcpy(cfg.d_population, cfg.population, pop_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cfg.d_distances, cfg.distances, dist_bytes, cudaMemcpyHostToDevice);

    // Copy cities to device
    City* d_cities = nullptr;
    cudaMalloc(&d_cities, (size_t)cfg.n_cities * sizeof(City));
    cudaMemcpy(d_cities, cfg.cities, (size_t)cfg.n_cities * sizeof(City), cudaMemcpyHostToDevice);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < cfg.n_generations; ++gen) {
        // Compute raw fitness and per-block partial sums on GPU
        int blockSize = 256;
        int gridSize = (cfg.n_population + blockSize - 1) / blockSize;

        size_t prob_bytes = (size_t)(cfg.n_population + gridSize) * sizeof(double);
        double* d_probs = nullptr;
        cudaMalloc(&d_probs, prob_bytes);
        cudaMemset(d_probs, 0, prob_bytes);

        ga2opt_compute_fitness_prob_device<<<gridSize, blockSize, blockSize * sizeof(double)>>>(cfg, cfg.d_distances, d_probs);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[GPU-GA+2opt] fitness kernel failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        // Fetch per-block partial sums and reduce on host to get total fitness
        double* h_block_sums = new double[gridSize];
        cudaMemcpy(h_block_sums, d_probs + cfg.n_population, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
        double total_fitness = 0.0;
        for (int b = 0; b < gridSize; ++b) total_fitness += h_block_sums[b];
        delete[] h_block_sums;
        ga2opt_normalize_probs_device<<<gridSize, blockSize>>>(cfg, d_probs, total_fitness);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[GPU-GA+2opt] normalize kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_probs);
            break;
        }

        // Elitism: do it on device by copying best individual index from host
        cudaMemcpy(cfg.distances, cfg.d_distances, dist_bytes, cudaMemcpyDeviceToHost);
        int best_idx = ga2opt_find_best_individual(cfg);
        // copy elite chromosome device->device into d_next_population[0]
        cudaMemcpy(cfg.d_next_population,
                   cfg.d_population + best_idx * cfg.n_cities,
                   (size_t)cfg.n_cities * sizeof(int),
                   cudaMemcpyDeviceToDevice);

        // also copy its distance
        cudaMemcpy(cfg.d_next_distances,
                   cfg.d_distances + best_idx,
                   sizeof(double),
                   cudaMemcpyDeviceToDevice);

        // Launch kernel to generate remaining offspring entirely on GPU
        unsigned int seed = (unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        int blockSizeOffspring = 256;
        int n_units = (cfg.n_population - 1 + 1) / 2; // ceil((n_population-1)/2)
        int gridSizeOffspring = (n_units + blockSizeOffspring - 1) / blockSizeOffspring;
        ga2opt_cuda_kernel<<<gridSizeOffspring, blockSizeOffspring>>>(cfg, d_probs, d_cities, seed);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[GPU-GA+2opt] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_probs);
            break;
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[GPU-GA+2opt] Kernel sync failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_probs);
            break;
        }

        cudaFree(d_probs);

        // Swap device populations/distances (keep host copies only for logging)
        int* tmp_d_pop = cfg.d_population;
        cfg.d_population = cfg.d_next_population;
        cfg.d_next_population = tmp_d_pop;

        double* tmp_d_dist = cfg.d_distances;
        cfg.d_distances = cfg.d_next_distances;
        cfg.d_next_distances = tmp_d_dist;

        // Every 100 generations (and last), pull distances back and log best
        if (gen % 100 == 0 || gen == cfg.n_generations - 1) {
            cudaMemcpy(cfg.distances, cfg.d_distances, dist_bytes, cudaMemcpyDeviceToHost);
            int current_best_idx = ga2opt_find_best_individual(cfg);
            double best_dist = cfg.distances[current_best_idx];
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            start_time = current_time;
            std::cout << "[GPU-GA+2opt] Generation " << gen
                      << " Best Distance: " << best_dist
                      << " Time: " << elapsed.count() << "s" << std::endl;
        }
    }

    // Final result: copy full population and distances back to host, report best
    cudaMemcpy(cfg.population, cfg.d_population, pop_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cfg.distances, cfg.d_distances, dist_bytes, cudaMemcpyDeviceToHost);

    int final_best_idx = ga2opt_find_best_individual(cfg);
    std::cout << "[GPU-GA+2opt] Final Best Distance: " << cfg.distances[final_best_idx] << std::endl;
    std::cout << "Best Route: ";
    int* best_chrom = cfg.population + final_best_idx * cfg.n_cities;
    for (int i = 0; i < cfg.n_cities; ++i) {
        std::cout << best_chrom[i] << " ";
    }
    std::cout << best_chrom[0] << std::endl;

    cudaFree(cfg.d_population);
    cudaFree(cfg.d_next_population);
    cudaFree(cfg.d_distances);
    cudaFree(cfg.d_next_distances);
}

int main(int argc, char** argv) {
    const std::string dataset = (argc > 1) ? argv[1] : "dj38.tsp";
    const int n_population = 4096;   // For cuda, set to a power of two
    const int n_generations = 200; // You can tune these values as needed
    const double crossover_rate = 0.8;
    const double mutation_rate = 0.2;
    const double init_two_opt_prob = 1.0;         // probability to refine an initial individual
    const double offspring_two_opt_prob = 0.5;    // probability to refine a child each generation
    const int two_opt_passes_init = 100;          // cap 2-opt passes for initial population
    const int two_opt_passes_offspring = 50;     // cap 2-opt passes for offspring

    srand(time(NULL));

    try {
        Dataloader dl(dataset);
        int n_cities = dl.cities.size();
        City* cities = new City[n_cities];
        for(int i=0; i<n_cities; ++i) cities[i] = dl.cities[i];

        GA2OptConfig cfg;
        cfg.cities = cities;
        cfg.n_cities = n_cities;
        cfg.n_population = n_population;
        cfg.n_generations = n_generations;
        cfg.crossover_rate = crossover_rate;
        cfg.mutation_rate = mutation_rate;
        cfg.init_two_opt_prob = init_two_opt_prob;
        cfg.offspring_two_opt_prob = offspring_two_opt_prob;
        cfg.two_opt_passes_init = two_opt_passes_init;
        cfg.two_opt_passes_offspring = two_opt_passes_offspring;

        cfg.population = new int[cfg.n_population * cfg.n_cities];
        cfg.distances = new double[cfg.n_population];
        cfg.next_population = new int[cfg.n_population * cfg.n_cities];
        cfg.next_distances = new double[cfg.n_population];

        ga2opt_initial_population(cfg);

        std::cout << "Starting GA + 2-opt for TSP (" << dataset << ")..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        ga2opt_run(cfg);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time: " << elapsed.count() << " s" << std::endl;
        
        delete[] cities;
        delete[] cfg.population;
        delete[] cfg.distances;
        delete[] cfg.next_population;
        delete[] cfg.next_distances;
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load dataset '" << dataset << "': " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}