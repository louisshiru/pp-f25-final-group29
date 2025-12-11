#ifndef KERNEL_H
#define KERNEL_H

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
    int* d_population;
    int* d_next_population;
    double* d_distances;
    double* d_next_distances;
    City* d_cities;
    // Precomputed dense distance matrix on device (row-major n_cities x n_cities)
    double* d_dist_matrix;
    double* d_probs;

    int* d_best_idx;
    double* d_total_fitness;
    // Nearest-neighbor candidate list (device): linearized as [n_cities][n_neighbors]
    int n_neighbors;
    int* d_neighbors;
};

// Host front-end for GPU 2-opt refinement; operates on raw arrays.
void host_fe_ga_2opt(int two_opt_passes_offspring,
                     double offspring_two_opt_prob);

// Host front-ends for GA crossover + mutation on GPU.
// They take the current population and distances, and produce the
// next generation (including elitism at index 0).
void host_fe_ga_selection_crossover_mutate(double crossover_rate,
                          double mutation_rate,
                          int two_opt_passes_offspring,
                          double offspring_two_opt_prob);

// Allocate / free persistent device buffers used by GA+2opt.
// These must be called before / after any GA CUDA calls.
void host_fe_ga_allocate(const City* cities,
                         int n_cities,
                         int n_population);
void host_fe_ga_free();

// Explicit host↔device population copy helpers.
void host_fe_ga_copy_population_to_device(const int* population,
                                          const double* distances);
void host_fe_ga_copy_population_to_host(int* population,
                                        double* distances);

#endif /* KERNEL_H */
