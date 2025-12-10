#include "dataloader.h" 
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <cstring>

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

std::vector<int> flatten_population(const std::vector<Individual>& pop, int num_cities) {
    std::vector<int> flat;
    flat.reserve(pop.size() * num_cities);
    for (const auto& ind : pop) {
        flat.insert(flat.end(), ind.chromosome.begin(), ind.chromosome.end());
    }
    return flat;
}

std::vector<Individual> reconstruct_population(const std::vector<int>& flat, int num_cities, const DistanceMatrix& dist_matrix) {
    if (flat.empty()) return {};
    int num_inds = flat.size() / num_cities;
    std::vector<Individual> pop;
    pop.reserve(num_inds);

    for (int i = 0; i < num_inds; ++i) {
        std::vector<int> chrom(flat.begin() + i * num_cities, flat.begin() + (i + 1) * num_cities);
        double d = total_distance(chrom, dist_matrix);
        pop.push_back({chrom, d});
    }
    return pop;
}

class GAWorker {
public:
    GAWorker(const std::vector<City>& cities, double crossover_rate, double mutation_rate, 
             double init_two_opt_prob, double offspring_two_opt_prob, int two_opt_passes_init, int two_opt_passes_offspring, int rank)
        : cities_(cities),
          dist_matrix_(cities),
          crossover_rate_(crossover_rate),
          mutation_rate_(mutation_rate),
          init_two_opt_prob_(init_two_opt_prob),
          offspring_two_opt_prob_(offspring_two_opt_prob),
          two_opt_passes_init_(two_opt_passes_init),
          two_opt_passes_offspring_(two_opt_passes_offspring),
          rng_(rank + 1) {}

    // 只有 Rank 0 會呼叫(產生初始族群)
    std::vector<Individual> create_initial_population(int total_pop_size) {
        std::vector<Individual> pop;
        std::vector<int> base(cities_.size());
        std::iota(base.begin(), base.end(), 0);

        pop.reserve(total_pop_size);
        for (int i = 0; i < total_pop_size; ++i) {
            std::vector<int> chrom = base;
            std::shuffle(chrom.begin(), chrom.end(), rng_);
            if (random_real() < init_two_opt_prob_) {
                chrom = two_opt(chrom, dist_matrix_, two_opt_passes_init_);
            }
            const double dist = total_distance(chrom, dist_matrix_);
            pop.push_back({chrom, dist});
        }
        return pop;
    }

    // 使用「全域父母」產生「本地子代」
    std::vector<Individual> generate_offspring_batch(const std::vector<Individual>& global_parents, int num_offspring_needed) {
        const std::vector<double> probs = fitness_prob(global_parents);
        std::vector<Individual> offspring_batch;
        offspring_batch.reserve(num_offspring_needed);
        
        while (static_cast<int>(offspring_batch.size()) < num_offspring_needed) {
            const Individual p1 = roulette_wheel(global_parents, probs);
            const Individual p2 = roulette_wheel(global_parents, probs);

            if (random_real() < crossover_rate_) {
                auto children = crossover(p1, p2);
                append_child(offspring_batch, children.first, num_offspring_needed);
                if (static_cast<int>(offspring_batch.size()) < num_offspring_needed) {
                    append_child(offspring_batch, children.second, num_offspring_needed);
                }
            } else {
                append_child(offspring_batch, p1, num_offspring_needed);
                if (static_cast<int>(offspring_batch.size()) < num_offspring_needed) {
                    append_child(offspring_batch, p2, num_offspring_needed);
                }
            }
        }
        return offspring_batch;
    }

    const DistanceMatrix& get_dist_matrix() const { return dist_matrix_; }

private:
    std::vector<City> cities_;
    DistanceMatrix dist_matrix_;
    double crossover_rate_;
    double mutation_rate_;
    double init_two_opt_prob_;
    double offspring_two_opt_prob_;
    int two_opt_passes_init_;
    int two_opt_passes_offspring_;
    std::mt19937 rng_;

    double random_real() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng_);
    }

    int random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }

    std::vector<double> fitness_prob(const std::vector<Individual>& pop) {
        std::vector<double> fitness_vals;
        fitness_vals.reserve(pop.size());
        double total_fitness = 0.0;
        for (const auto& ind : pop) {
            const double f = 1.0 / ind.distance;
            fitness_vals.push_back(f);
            total_fitness += f;
        }
        std::vector<double> probs;
        probs.reserve(pop.size());
        for (double f : fitness_vals) {
            probs.push_back(f / total_fitness);
        }
        return probs;
    }

    Individual roulette_wheel(const std::vector<Individual>& pop, const std::vector<double>& probs) {
        const double r = random_real();
        double cum_prob = 0.0;
        for (size_t i = 0; i < pop.size(); ++i) {
            cum_prob += probs[i];
            if (r <= cum_prob) {
                return pop[i];
            }
        }
        return pop.back();
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
        return {{c1, 0.0}, {c2, 0.0}}; 
    }

    void mutate(std::vector<int>& chrom) {
        if (random_real() < mutation_rate_) {
            const int n = static_cast<int>(chrom.size());
            const int i = random_int(0, n - 1);
            const int j = random_int(0, n - 1);
            std::swap(chrom[i], chrom[j]);
        }
    }

    void append_child(std::vector<Individual>& pop, Individual child, int limit) {
        if (static_cast<int>(pop.size()) >= limit) return;
        
        mutate(child.chromosome);
        // 2-Opt on offspring
        if (random_real() < offspring_two_opt_prob_) {
            child.chromosome = two_opt(child.chromosome, dist_matrix_, two_opt_passes_offspring_);
        }
        child.distance = total_distance(child.chromosome, dist_matrix_);
        pop.push_back(std::move(child));
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string dataset = (argc > 1) ? argv[1] : "qa194.tsp";
    
 
    const int total_population = 8192;
    const int local_offspring_count = total_population / size;
    const int n_generations = 3000;
    
    // GA 參數
    const double crossover_rate = 0.8;
    const double mutation_rate = 0.2;
    const double init_two_opt_prob = 1.0;         
    const double offspring_two_opt_prob = 1.0;    
    const int two_opt_passes_init = 100;          
    const int two_opt_passes_offspring = 2;     

    try {
        Dataloader dl(dataset);
        int num_cities = dl.cities.size();

        GAWorker worker(dl.cities, crossover_rate, mutation_rate, 
                         init_two_opt_prob, offspring_two_opt_prob, two_opt_passes_init, two_opt_passes_offspring, rank);

        std::vector<int> global_pop_flat; 
        global_pop_flat.resize(total_population * num_cities);

        std::vector<int> local_offspring_flat; 
        
        if (rank == 0) {
            std::cout << "Starting MPI GA (Broadcast Model) for TSP (" << dataset << ")..." << std::endl;
            std::vector<Individual> init_pop = worker.create_initial_population(total_population);
            std::vector<int> temp_flat = flatten_population(init_pop, num_cities);
            std::copy(temp_flat.begin(), temp_flat.end(), global_pop_flat.begin());
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (int gen = 0; gen < n_generations; ++gen) {
            MPI_Bcast(global_pop_flat.data(), total_population * num_cities, MPI_INT, 0, MPI_COMM_WORLD);

            std::vector<Individual> global_objs = reconstruct_population(global_pop_flat, num_cities, worker.get_dist_matrix());
            Individual best_parent; 
            if (rank == 0) {
                auto it = std::min_element(global_objs.begin(), global_objs.end(), 
                    [](const Individual& a, const Individual& b) {
                        return a.distance < b.distance;
                    });
                best_parent = *it;
                if (gen % 100 == 0) {
                     std::cout << "Generation " << gen << " Best: " << best_parent.distance << std::endl;
                }
            }

            std::vector<Individual> local_new_objs = worker.generate_offspring_batch(global_objs, local_offspring_count);
            local_offspring_flat = flatten_population(local_new_objs, num_cities);

            MPI_Gather(local_offspring_flat.data(), local_offspring_count * num_cities, MPI_INT,
                       global_pop_flat.data(), local_offspring_count * num_cities, MPI_INT,
                       0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                std::copy(best_parent.chromosome.begin(), best_parent.chromosome.end(), global_pop_flat.begin());
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            std::vector<Individual> final_pop = reconstruct_population(global_pop_flat, num_cities, worker.get_dist_matrix());
            
            auto best_it = std::min_element(
                final_pop.begin(), final_pop.end(),
                [](const Individual& a, const Individual& b) { return a.distance < b.distance; });

            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Time: " << elapsed.count() << " s" << std::endl;
            std::cout << "Final Best Distance: " << best_it->distance << std::endl;
        }

    } catch (const std::exception& ex) {
        std::cerr << "Rank " << rank << " Error: " << ex.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}