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

class GA2Opt {
public:
    GA2Opt(const City* cities, int n_cities, int population, int generations, double crossover_rate, double mutation_rate, double init_two_opt_prob = 0.25,
           double offspring_two_opt_prob = 0.15, int two_opt_passes_init = 3, int two_opt_passes_offspring = 2)
        : cities_(cities),
          n_cities_(n_cities),
          n_population_(population),
          n_generations_(generations),
          crossover_rate_(crossover_rate),
          mutation_rate_(mutation_rate),
          init_two_opt_prob_(init_two_opt_prob),
          offspring_two_opt_prob_(offspring_two_opt_prob),
          two_opt_passes_init_(two_opt_passes_init),
          two_opt_passes_offspring_(two_opt_passes_offspring) {
        
        // Allocate memory
        population_ = new int[n_population_ * n_cities_];
        distances_ = new double[n_population_];
        next_population_ = new int[n_population_ * n_cities_];
        next_distances_ = new double[n_population_];
        
        // Initialize population
        initial_population();
    }

    ~GA2Opt() {
        delete[] population_;
        delete[] distances_;
        delete[] next_population_;
        delete[] next_distances_;
    }

    void run() {
        double* probs = new double[n_population_];
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int gen = 0; gen < n_generations_; ++gen) {
            compute_fitness_prob(probs);

            int next_pop_count = 0;

            // Elitism: keep the current best individual.
            int best_idx = find_best_individual();
            
            // Copy best to next population
            copy_individual(population_ + best_idx * n_cities_, next_population_ + next_pop_count * n_cities_);
            next_distances_[next_pop_count] = distances_[best_idx];
            next_pop_count++;

            while (next_pop_count < n_population_) {
                int p1_idx = roulette_wheel(probs);
                int p2_idx = roulette_wheel(probs);

                int* child1_ptr = next_population_ + next_pop_count * n_cities_;
                // Check if we have space for second child
                int* child2_ptr = (next_pop_count + 1 < n_population_) ? next_population_ + (next_pop_count + 1) * n_cities_ : nullptr;

                if (random_real() < crossover_rate_) {
                    crossover(population_ + p1_idx * n_cities_, population_ + p2_idx * n_cities_, child1_ptr, child2_ptr);
                    
                    process_child(child1_ptr);
                    next_distances_[next_pop_count] = total_distance(child1_ptr, n_cities_, cities_);
                    next_pop_count++;
                    
                    if (child2_ptr && next_pop_count < n_population_) {
                        process_child(child2_ptr);
                        next_distances_[next_pop_count] = total_distance(child2_ptr, n_cities_, cities_);
                        next_pop_count++;
                    }
                } else {
                    copy_individual(population_ + p1_idx * n_cities_, child1_ptr);
                    process_child(child1_ptr);
                    next_distances_[next_pop_count] = total_distance(child1_ptr, n_cities_, cities_);
                    next_pop_count++;

                    if (child2_ptr && next_pop_count < n_population_) {
                        copy_individual(population_ + p2_idx * n_cities_, child2_ptr);
                        process_child(child2_ptr);
                        next_distances_[next_pop_count] = total_distance(child2_ptr, n_cities_, cities_);
                        next_pop_count++;
                    }
                }
            }

            // Swap populations
            int* temp_pop = population_;
            population_ = next_population_;
            next_population_ = temp_pop;

            double* temp_dist = distances_;
            distances_ = next_distances_;
            next_distances_ = temp_dist;

            int current_best_idx = find_best_individual();
            double best_dist = distances_[current_best_idx];

            if (gen % 100 == 0 || gen == n_generations_ - 1) {
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = current_time - start_time;
                start_time = current_time;
                std::cout << "Generation " << gen << " Best Distance: " << best_dist << " Time: " << elapsed.count() << "s" << std::endl;
            }
        }

        int final_best_idx = find_best_individual();
        std::cout << "Final Best Distance: " << distances_[final_best_idx] << std::endl;
        std::cout << "Best Route: ";
        int* best_chrom = population_ + final_best_idx * n_cities_;
        for (int i = 0; i < n_cities_; ++i) {
            std::cout << best_chrom[i] << " ";
        }
        std::cout << best_chrom[0] << std::endl; // close the tour
        
        delete[] probs;
    }

private:
    const City* cities_;
    int n_cities_;
    int n_population_;
    int n_generations_;
    double crossover_rate_;
    double mutation_rate_;
    double init_two_opt_prob_;
    double offspring_two_opt_prob_;
    int two_opt_passes_init_;
    int two_opt_passes_offspring_;
    
    int* population_;
    double* distances_;
    int* next_population_;
    double* next_distances_;

    double random_real() {
        return (double)rand() / RAND_MAX;
    }

    int random_int(int min, int max) {
        return min + rand() % (max - min + 1);
    }

    void shuffle_array(int* arr, int n) {
        for (int i = n - 1; i > 0; i--) {
            int j = random_int(0, i);
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    void initial_population() {
        int* base = new int[n_cities_];
        for(int i=0; i<n_cities_; ++i) base[i] = i;

        for (int i = 0; i < n_population_; ++i) {
            int* chrom = population_ + i * n_cities_;
            // Copy base
            for(int j=0; j<n_cities_; ++j) chrom[j] = base[j];
            
            shuffle_array(chrom, n_cities_);
            
            if (random_real() < init_two_opt_prob_) {
                two_opt(chrom, n_cities_, cities_, two_opt_passes_init_);
            }
            distances_[i] = total_distance(chrom, n_cities_, cities_);
        }
        delete[] base;
    }

    void compute_fitness_prob(double* probs) {
        double total_fitness = 0.0;
        for (int i = 0; i < n_population_; ++i) {
            double f = 1.0 / distances_[i];
            probs[i] = f;
            total_fitness += f;
        }
        for (int i = 0; i < n_population_; ++i) {
            probs[i] /= total_fitness;
        }
    }

    int roulette_wheel(const double* probs) {
        double r = random_real();
        double cum_prob = 0.0;
        for (int i = 0; i < n_population_; ++i) {
            cum_prob += probs[i];
            if (r <= cum_prob) {
                return i;
            }
        }
        return n_population_ - 1;
    }

    int find_best_individual() {
        int best_idx = 0;
        for (int i = 1; i < n_population_; ++i) {
            if (distances_[i] < distances_[best_idx]) {
                best_idx = i;
            }
        }
        return best_idx;
    }

    void copy_individual(const int* src, int* dest) {
        for(int i=0; i<n_cities_; ++i) {
            dest[i] = src[i];
        }
    }

    void create_child_helper(const int* parent_a, const int* parent_b, int* child, int cut1, int cut2, bool* used) {
        for(int i=0; i<n_cities_; ++i) {
            child[i] = -1;
            used[i] = false;
        }

        for (int i = cut1; i <= cut2; ++i) {
            child[i] = parent_a[i];
            used[parent_a[i]] = true;
        }

        int idx_b = (cut2 + 1) % n_cities_;
        int idx_child = (cut2 + 1) % n_cities_;
        for (int i = 0; i < n_cities_; ++i) {
            int candidate = parent_b[idx_b];
            if (!used[candidate]) {
                child[idx_child] = candidate;
                used[candidate] = true;
                idx_child = (idx_child + 1) % n_cities_;
            }
            idx_b = (idx_b + 1) % n_cities_;
        }
    }

    void crossover(const int* p1, const int* p2, int* c1, int* c2) {
        int cut1 = random_int(0, n_cities_ - 1);
        int cut2 = random_int(0, n_cities_ - 1);
        if (cut1 > cut2) { int t = cut1; cut1 = cut2; cut2 = t; }

        bool* used = new bool[n_cities_];

        create_child_helper(p1, p2, c1, cut1, cut2, used);
        if (c2) {
            create_child_helper(p2, p1, c2, cut1, cut2, used);
        }

        delete[] used;
    }

    void mutate(int* chrom) {
        if (random_real() < mutation_rate_) {
            int i = random_int(0, n_cities_ - 1);
            int j = random_int(0, n_cities_ - 1);
            int temp = chrom[i];
            chrom[i] = chrom[j];
            chrom[j] = temp;
        }
    }

    void process_child(int* child) {
        mutate(child);
        if (random_real() < offspring_two_opt_prob_) {
            two_opt(child, n_cities_, cities_, two_opt_passes_offspring_);
        }
    }
};

int main(int argc, char** argv) {
    const std::string dataset = (argc > 1) ? argv[1] : "za929.tsp";
    const int n_population = 1000;   // Slightly smaller because 2-opt is heavier
    const int n_generations = 1000; // You can tune these values as needed
    const double crossover_rate = 0.8;
    const double mutation_rate = 0.2;
    const double init_two_opt_prob = 1.0;         // probability to refine an initial individual
    const double offspring_two_opt_prob = 0.5;    // probability to refine a child each generation
    const int two_opt_passes_init = 100;          // cap 2-opt passes for initial population
    const int two_opt_passes_offspring = 30;     // cap 2-opt passes for offspring

    srand(time(NULL));

    try {
        Dataloader dl(dataset);
        int n_cities = dl.cities.size();
        City* cities = new City[n_cities];
        for(int i=0; i<n_cities; ++i) cities[i] = dl.cities[i];

        GA2Opt solver(cities, n_cities, n_population, n_generations, crossover_rate, mutation_rate, 
                      init_two_opt_prob, offspring_two_opt_prob, two_opt_passes_init, two_opt_passes_offspring);

        std::cout << "Starting GA + 2-opt for TSP (" << dataset << ")..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        solver.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time: " << elapsed.count() << " s" << std::endl;
        
        delete[] cities;
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load dataset '" << dataset << "': " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}