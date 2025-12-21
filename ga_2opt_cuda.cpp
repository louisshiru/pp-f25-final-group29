#include "dataloader.h"
#include "kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

// Dense distance matrix keeps lookups cache-friendly for GA + 2-opt,
// so we only pay the sqrt cost once during construction.
class DistanceMatrix {
public:
    explicit DistanceMatrix(const std::vector<City>& cities)
        : n_(cities.size()), data_(n_ * n_, 0.0) {
        for (size_t i = 0; i < n_; ++i) {
            for (size_t j = i + 1; j < n_; ++j) {
                const double dx = cities[i].x - cities[j].x;
                const double dy = cities[i].y - cities[j].y;
                const double d = std::sqrt(dx * dx + dy * dy);
                data_[i * n_ + j] = d;
                data_[j * n_ + i] = d;
            }
        }
    }

    double operator()(int i, int j) const {
        return data_[static_cast<size_t>(i) * n_ + static_cast<size_t>(j)];
    }

    size_t size() const { return n_; }

private:
    size_t n_;
    std::vector<double> data_;
};

double total_distance(const std::vector<int>& route, const DistanceMatrix& dist) {
    double total = 0.0;
    const size_t n = route.size();
    for (size_t i = 0; i < n; ++i) {
        total += dist(route[i], route[(i + 1) % n]);
    }
    return total;
}

// A capped 2-opt: at most `max_passes` improvement rounds to avoid long runtimes on large instances.
std::vector<int> two_opt(std::vector<int> route, const DistanceMatrix& dist, int max_passes) {
    const size_t n = route.size();
    if (n < 4) return route;

    bool improved = true;
    int passes = 0;
    while (improved && passes < max_passes) {
        improved = false;
        for (size_t i = 1; i < n - 1 && !improved; ++i) {
            for (size_t k = i + 1; k < n && !improved; ++k) {
                const size_t next = (k + 1) % n;
                const double delta =
                    dist(route[i - 1], route[k]) +
                    dist(route[i], route[next]) -
                    dist(route[i - 1], route[i]) -
                    dist(route[k], route[next]);

                if (delta < -1e-9) {
                    std::reverse(route.begin() + static_cast<long>(i), route.begin() + static_cast<long>(k) + 1);
                    improved = true;
                }
            }
        }
        ++passes;
    }

    return route;
}

struct Individual {
    std::vector<int> chromosome;
    double distance;
};

class GA2Opt {
public:
    GA2Opt(const std::vector<City>& cities, int population, int generations, double crossover_rate, double mutation_rate, double init_two_opt_prob = 0.25,
           double offspring_two_opt_prob = 0.15, int two_opt_passes_init = 3, int two_opt_passes_offspring = 2)
        : cities_(cities),
          dist_matrix_(cities),
          n_population_(population),
          n_generations_(generations),
          crossover_rate_(crossover_rate),
          mutation_rate_(mutation_rate),
          init_two_opt_prob_(init_two_opt_prob),         // probability to refine an initial individual
          offspring_two_opt_prob_(offspring_two_opt_prob),    // probability to refine a child each generation
          two_opt_passes_init_(two_opt_passes_init),          // cap 2-opt passes for initial population
          two_opt_passes_offspring_(two_opt_passes_offspring),     // cap 2-opt passes for offspring
          rng_(std::random_device{}()) {
        population_ = initial_population();
    }

    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();
        // Flattened buffers reused across generations (host-side snapshots)
        std::vector<int> flat_pop(static_cast<size_t>(n_population_) * cities_.size());
        std::vector<double> flat_dist(static_cast<size_t>(n_population_));
        
        // Initialize device resources and upload initial population once
        host_fe_ga_allocate(cities_.data(), static_cast<int>(cities_.size()), n_population_);
        for (int i = 0; i < n_population_; ++i) {
            const size_t offset = static_cast<size_t>(i) * cities_.size();
            std::copy(population_[i].chromosome.begin(), population_[i].chromosome.end(),
                      flat_pop.begin() + static_cast<long>(offset));
            flat_dist[static_cast<size_t>(i)] = population_[i].distance;
        }
        host_fe_ga_copy_population_to_device(flat_pop.data(), flat_dist.data());

        for (int gen = 0; gen < n_generations_; ++gen) {

            // GA step (selection + crossover + mutation) on GPU, operating purely on device buffers
            host_fe_ga_selection_crossover_mutate(crossover_rate_,
                                 mutation_rate_,
                                 two_opt_passes_offspring_,
                                 offspring_two_opt_prob_);

            // Optional extra 2-opt refinement pass on the offspring (GPU / CUDA)
            host_fe_ga_2opt(two_opt_passes_offspring_, offspring_two_opt_prob_);

            if (gen % 1000 == 0 || gen == n_generations_ - 1) {
                // Snapshot population from device only when needed for reporting
                host_fe_ga_copy_population_to_host(flat_pop.data(), flat_dist.data());
                const double best_dist = *std::min_element(flat_dist.begin(), flat_dist.end());
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = current_time - start_time;
                start_time = current_time;
                std::cout << "Generation " << gen << " Best Distance: " << best_dist << " Time: " << elapsed.count() << "s" << std::endl;
            }
        }
        // Final snapshot from device for reporting the best tour
        host_fe_ga_copy_population_to_host(flat_pop.data(), flat_dist.data());
        host_fe_ga_free();

        const auto best_it = std::min_element(flat_dist.begin(), flat_dist.end());
        const int best_idx = static_cast<int>(std::distance(flat_dist.begin(), best_it));
        const size_t route_offset = static_cast<size_t>(best_idx) * cities_.size();

        std::cout << "Final Best Distance: " << *best_it << std::endl;
        std::cout << "Best Route: ";
        for (size_t i = 0; i < cities_.size(); ++i) {
            std::cout << flat_pop[route_offset + i] << " ";
        }
        // close the tour
        std::cout << flat_pop[route_offset] << std::endl;
    }

private:
    std::vector<City> cities_;
    DistanceMatrix dist_matrix_;
    int n_population_;
    int n_generations_;
    double crossover_rate_;
    double mutation_rate_;
    double init_two_opt_prob_;
    double offspring_two_opt_prob_;
    int two_opt_passes_init_;
    int two_opt_passes_offspring_;
    std::vector<Individual> population_;
    std::mt19937 rng_;

    double random_real() {
        static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng_);
    }

    int random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }

    std::vector<Individual> initial_population() {
        std::vector<Individual> pop;
        std::vector<int> base(cities_.size());
        std::iota(base.begin(), base.end(), 0);

        pop.reserve(n_population_);
        for (int i = 0; i < n_population_; ++i) {
            std::vector<int> chrom = base;
            std::shuffle(chrom.begin(), chrom.end(), rng_);
            // Randomly decide whether to run 2-opt on this individual to keep runtime manageable.
            if (random_real() < init_two_opt_prob_) {
                chrom = two_opt(chrom, dist_matrix_, two_opt_passes_init_);
            }
            const double dist = total_distance(chrom, dist_matrix_);
            pop.push_back({chrom, dist});
        }
        return pop;
    }

    std::vector<double> fitness_cumulative(const std::vector<Individual>& pop) {
        std::vector<double> cumulative(pop.size(), 0.0);
        double total_fitness = 0.0;
        for (const auto& ind : pop) {
            total_fitness += 1.0 / ind.distance;
        }
        double running = 0.0;
        for (size_t i = 0; i < pop.size(); ++i) {
            running += (1.0 / pop[i].distance) / total_fitness;
            cumulative[i] = running;
        }
        if (!cumulative.empty()) {
            cumulative.back() = 1.0; // avoid floating-point gap at the end
        }
        return cumulative;
    }

    Individual roulette_wheel(const std::vector<Individual>& pop, const std::vector<double>& cumulative) {
        const double r = random_real();
        const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        const size_t idx = static_cast<size_t>(std::distance(cumulative.begin(), (it == cumulative.end()) ? cumulative.end() - 1 : it));
        return pop[idx];
    }

    std::pair<Individual, Individual> crossover(const Individual& p1, const Individual& p2) {
        const int n = static_cast<int>(p1.chromosome.size());
        int cut1 = random_int(0, n - 1);
        int cut2 = random_int(0, n - 1);
        if (cut1 > cut2) std::swap(cut1, cut2);

        auto create_child = [&](const std::vector<int>& a, const std::vector<int>& b) {
            std::vector<int> child(n, -1);
            std::vector<bool> used(n, false);

            for (int i = cut1; i <= cut2; ++i) {
                child[i] = a[i];
                used[a[i]] = true;
            }

            int idx_b = (cut2 + 1) % n;
            int idx_child = (cut2 + 1) % n;
            for (int i = 0; i < n; ++i) {
                const int candidate = b[idx_b];
                if (!used[candidate]) {
                    child[idx_child] = candidate;
                    used[candidate] = true;
                    idx_child = (idx_child + 1) % n;
                }
                idx_b = (idx_b + 1) % n;
            }

            return child;
        };

        std::vector<int> c1 = create_child(p1.chromosome, p2.chromosome);
        std::vector<int> c2 = create_child(p2.chromosome, p1.chromosome);
        return {{c1, 0.0}, {c2, 0.0}}; // distance will be set later
    }

    void mutate(std::vector<int>& chrom) {
        if (random_real() < mutation_rate_) {
            const int n = static_cast<int>(chrom.size());
            const int i = random_int(0, n - 1);
            const int j = random_int(0, n - 1);
            std::swap(chrom[i], chrom[j]);
        }
    }

    Individual mutate(Individual child) {
        mutate(child.chromosome);
        child.distance = total_distance(child.chromosome, dist_matrix_);
        return child;
    }
};

int main(int argc, char** argv) {
    const std::string dataset = (argc > 1) ? argv[1] : "qa194.tsp";
    const int n_population = 8192;   // Slightly smaller because 2-opt is heavier
    const int n_generations = 3000; // You can tune these values as needed
    const double crossover_rate = 0.8;
    const double mutation_rate = 0.2;
    const double init_two_opt_prob = 1.0;         // probability to refine an initial individual
    const double offspring_two_opt_prob = 1.0;    // probability to refine a child each generation
    const int two_opt_passes_init = 100;          // cap 2-opt passes for initial population
    const int two_opt_passes_offspring = 2;     // cap 2-opt passes for offspring

    try {
        Dataloader dl(dataset);
        GA2Opt solver(dl.cities, n_population, n_generations, crossover_rate, mutation_rate, 
                      init_two_opt_prob, offspring_two_opt_prob, two_opt_passes_init, two_opt_passes_offspring);

        std::cout << "Starting GA + 2-opt for TSP (" << dataset << ")..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        solver.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time: " << elapsed.count() << " s" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load dataset '" << dataset << "': " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
