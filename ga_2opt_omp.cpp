#include "dataloader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Dense distance matrix keeps lookups cache-friendly for GA + 2-opt.
class DistanceMatrix {
public:
    explicit DistanceMatrix(const std::vector<City>& cities) : n_(cities.size()), data_(n_ * n_, 0.0) {
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

    double operator()(int i, int j) const { return data_[static_cast<size_t>(i) * n_ + static_cast<size_t>(j)]; }
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
                const double delta = dist(route[i - 1], route[k]) + dist(route[i], route[next]) -
                                     dist(route[i - 1], route[i]) - dist(route[k], route[next]);

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

class GA2OptOmp {
public:
    GA2OptOmp(const std::vector<City>& cities, int population, int generations, double crossover_rate, double mutation_rate,
              double init_two_opt_prob = 0.25, double offspring_two_opt_prob = 0.15, int two_opt_passes_init = 3,
              int two_opt_passes_offspring = 2)
        : cities_(cities),
          dist_matrix_(cities),
          n_population_(population),
          n_generations_(generations),
          crossover_rate_(crossover_rate),
          mutation_rate_(mutation_rate),
          init_two_opt_prob_(init_two_opt_prob),
          offspring_two_opt_prob_(offspring_two_opt_prob),
          two_opt_passes_init_(two_opt_passes_init),
          two_opt_passes_offspring_(two_opt_passes_offspring),
          base_seed_(static_cast<unsigned int>(
              std::chrono::high_resolution_clock::now().time_since_epoch().count())) {
        setup_rngs();
        population_ = initial_population();
    }

    void run() {
        // Reuse buffers across generations to reduce allocations.
        std::vector<double> cumulative_probs;
        std::vector<Individual> next_population;

#pragma omp parallel
        {
            for (int gen = 0; gen < n_generations_; ++gen) {
#pragma omp single
                {
                    cumulative_probs = fitness_cumulative(population_);
                    next_population.assign(static_cast<size_t>(n_population_), Individual{});

                    // Elitism: keep the current best individual.
                    const auto best_it = std::min_element(
                        population_.begin(), population_.end(),
                        [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
                    next_population[0] = *best_it;
                }

                // Parallel offspring generation: each thread works on disjoint slots to avoid contention.
#pragma omp for schedule(dynamic, 2)
                for (int idx = 1; idx < n_population_; idx += 2) {
                    std::mt19937& rng = thread_rng();

                    const Individual p1 = roulette_wheel(population_, cumulative_probs, rng);
                    const Individual p2 = roulette_wheel(population_, cumulative_probs, rng);

                    std::vector<int> child1_route;
                    std::vector<int> child2_route;
                    if (random_real(rng) < crossover_rate_) {
                        auto children = crossover(p1.chromosome, p2.chromosome, rng);
                        child1_route = std::move(children.first);
                        child2_route = std::move(children.second);
                    } else {
                        child1_route = p1.chromosome;
                        child2_route = p2.chromosome;
                    }

                    Individual child1{std::move(child1_route), 0.0};
                    child1 = mutate_and_local_search(std::move(child1), rng, offspring_two_opt_prob_, two_opt_passes_offspring_);
                    next_population[idx] = std::move(child1);

                    if (idx + 1 < n_population_) {
                        Individual child2{std::move(child2_route), 0.0};
                        child2 = mutate_and_local_search(std::move(child2), rng, offspring_two_opt_prob_, two_opt_passes_offspring_);
                        next_population[idx + 1] = std::move(child2);
                    }
                }

#pragma omp single
                {
                    population_.swap(next_population);

                    const double best_dist = std::min_element(
                                                 population_.begin(), population_.end(),
                                                 [](const Individual& a, const Individual& b) { return a.distance < b.distance; })
                                                 ->distance;

                    if (gen % 100 == 0 || gen == n_generations_ - 1) {
                        std::cout << "Generation " << gen << " Best Distance: " << best_dist << std::endl;
                    }
                }
            }
        }

        const auto best_it = std::min_element(
            population_.begin(), population_.end(),
            [](const Individual& a, const Individual& b) { return a.distance < b.distance; });

        std::cout << "Final Best Distance: " << best_it->distance << std::endl;
        std::cout << "Best Route: ";
        for (int idx : best_it->chromosome) {
            std::cout << idx << " ";
        }
        std::cout << best_it->chromosome.front() << std::endl; // close the tour
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
    std::vector<std::mt19937> rng_pool_;
    unsigned int base_seed_;

    void setup_rngs() {
        const int threads = std::max(1, omp_get_max_threads());
        rng_pool_.resize(static_cast<size_t>(threads));
        for (int t = 0; t < threads; ++t) {
            rng_pool_[static_cast<size_t>(t)].seed(base_seed_ + static_cast<unsigned int>(t * 9973));
        }
    }

    std::mt19937& thread_rng() {
        const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        const size_t idx = static_cast<size_t>(tid % static_cast<int>(rng_pool_.size()));
        return rng_pool_[idx];
    }

    double random_real(std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng);
    }

    int random_int(int min, int max, std::mt19937& rng) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng);
    }

    std::vector<Individual> initial_population() {
        std::vector<int> base(cities_.size());
        std::iota(base.begin(), base.end(), 0);

        std::vector<Individual> pop(static_cast<size_t>(n_population_));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_population_; ++i) {
            std::mt19937& rng = thread_rng();
            std::vector<int> chrom = base;
            std::shuffle(chrom.begin(), chrom.end(), rng);
            if (random_real(rng) < init_two_opt_prob_) {
                chrom = two_opt(std::move(chrom), dist_matrix_, two_opt_passes_init_);
            }
            const double dist = total_distance(chrom, dist_matrix_);
            pop[static_cast<size_t>(i)] = {std::move(chrom), dist};
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
        // Avoid edge cases due to floating point rounding.
        if (!cumulative.empty()) {
            cumulative.back() = 1.0;
        }
        return cumulative;
    }

    Individual roulette_wheel(const std::vector<Individual>& pop, const std::vector<double>& cumulative, std::mt19937& rng) {
        const double r = random_real(rng);
        const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        const size_t idx = static_cast<size_t>(std::distance(cumulative.begin(), (it == cumulative.end()) ? cumulative.end() - 1 : it));
        return pop[idx];
    }

    std::pair<std::vector<int>, std::vector<int>> crossover(const std::vector<int>& a, const std::vector<int>& b, std::mt19937& rng) {
        const int n = static_cast<int>(a.size());
        int cut1 = random_int(0, n - 1, rng);
        int cut2 = random_int(0, n - 1, rng);
        if (cut1 > cut2) std::swap(cut1, cut2);

        auto create_child = [&](const std::vector<int>& p1, const std::vector<int>& p2) {
            std::vector<int> child(n, -1);
            std::vector<bool> used(static_cast<size_t>(n), false);

            for (int i = cut1; i <= cut2; ++i) {
                child[i] = p1[i];
                used[static_cast<size_t>(p1[i])] = true;
            }

            int idx_b = (cut2 + 1) % n;
            int idx_child = (cut2 + 1) % n;
            for (int i = 0; i < n; ++i) {
                const int candidate = p2[idx_b];
                if (!used[static_cast<size_t>(candidate)]) {
                    child[idx_child] = candidate;
                    used[static_cast<size_t>(candidate)] = true;
                    idx_child = (idx_child + 1) % n;
                }
                idx_b = (idx_b + 1) % n;
            }

            return child;
        };

        std::vector<int> c1 = create_child(a, b);
        std::vector<int> c2 = create_child(b, a);
        return {std::move(c1), std::move(c2)};
    }

    void mutate(std::vector<int>& chrom, std::mt19937& rng) {
        if (random_real(rng) < mutation_rate_) {
            const int n = static_cast<int>(chrom.size());
            int i = random_int(0, n - 1, rng);
            int j = random_int(0, n - 1, rng);
            while (j == i && n > 1) {
                j = random_int(0, n - 1, rng);
            }
            std::swap(chrom[i], chrom[j]);
        }
    }

    Individual mutate_and_local_search(Individual child, std::mt19937& rng, double two_opt_prob, int max_passes) {
        mutate(child.chromosome, rng);
        if (random_real(rng) < two_opt_prob) {
            child.chromosome = two_opt(std::move(child.chromosome), dist_matrix_, max_passes);
        }
        child.distance = total_distance(child.chromosome, dist_matrix_);
        return child;
    }
};

int main(int argc, char** argv) {
    const std::string dataset = (argc > 1) ? argv[1] : "qa194.tsp";
    const int n_population = 8192;   // Slightly smaller because 2-opt is heavier
    // const int n_population = 100;   // Slightly smaller because 2-opt is heavier
    const int n_generations = 3000; // You can tune these values as needed
    const double crossover_rate = 0.8;
    const double mutation_rate = 0.2;
    const double init_two_opt_prob = 1.0;         // probability to refine an initial individual
    const double offspring_two_opt_prob = 1.0;    // probability to refine a child each generation
    // const double offspring_two_opt_prob = 0.1;    // probability to refine a child each generation
    const int two_opt_passes_init = 100;          // cap 2-opt passes for initial population
    const int two_opt_passes_offspring = 2;      // cap 2-opt passes for offspring
    // const int two_opt_passes_offspring = 50;      // cap 2-opt passes for offspring

    try {
        Dataloader dl(dataset);
        GA2OptOmp solver(dl.cities, n_population, n_generations, crossover_rate, mutation_rate, init_two_opt_prob,
                        offspring_two_opt_prob, two_opt_passes_init, two_opt_passes_offspring);

        std::cout << "Starting GA + 2-opt (OpenMP) for TSP (" << dataset << ")..." << std::endl;
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
