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

// --- DistanceMatrix & 2-Opt (保持不變) ---
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
    for (size_t i = 0; i < n; ++i) total += dist(route[i], route[(i + 1) % n]);
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
                const double delta = dist(route[i - 1], route[k]) + dist(route[i], route[next]) 
                                   - dist(route[i - 1], route[i]) - dist(route[k], route[next]);
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

// --- Flatten / Reconstruct (保持優化版) ---

void flatten_population(const std::vector<Individual>& pop, std::vector<int>& flat_paths, std::vector<double>& flat_dists) {
    size_t num_cities = pop[0].chromosome.size();
    if (flat_paths.size() != pop.size() * num_cities) flat_paths.resize(pop.size() * num_cities);
    if (flat_dists.size() != pop.size()) flat_dists.resize(pop.size());
    
    for (size_t i = 0; i < pop.size(); ++i) {
        std::copy(pop[i].chromosome.begin(), pop[i].chromosome.end(), flat_paths.begin() + i * num_cities);
        flat_dists[i] = pop[i].distance;
    }
}

void reconstruct_population(std::vector<Individual>& pop, const std::vector<int>& flat_paths, const std::vector<double>& flat_dists, int num_cities) {
    size_t num_inds = flat_dists.size();
    if (pop.size() != num_inds) pop.resize(num_inds); 

    for (size_t i = 0; i < num_inds; ++i) {
        if (pop[i].chromosome.size() != static_cast<size_t>(num_cities)) {
             pop[i].chromosome.resize(num_cities);
        }
        std::copy(flat_paths.begin() + i * num_cities, flat_paths.begin() + (i + 1) * num_cities, pop[i].chromosome.begin());
        pop[i].distance = flat_dists[i];
    }
}

// --- GA Worker Class (主要修改處) ---
class GAWorker {
public:
    GAWorker(const std::vector<City>& cities, double crossover_rate, double mutation_rate, 
             double init_two_opt_prob, double offspring_two_opt_prob, int two_opt_passes_init, int two_opt_passes_offspring, int rank)
        : cities_(cities), dist_matrix_(cities), crossover_rate_(crossover_rate), mutation_rate_(mutation_rate),
          init_two_opt_prob_(init_two_opt_prob), offspring_two_opt_prob_(offspring_two_opt_prob),
          two_opt_passes_init_(two_opt_passes_init), two_opt_passes_offspring_(two_opt_passes_offspring), rng_(rank + 1) {}

    std::vector<Individual> create_initial_population(int total_pop_size) {
        std::vector<Individual> pop;
        std::vector<int> base(cities_.size());
        std::iota(base.begin(), base.end(), 0);
        pop.reserve(total_pop_size);
        for (int i = 0; i < total_pop_size; ++i) {
            std::vector<int> chrom = base;
            std::shuffle(chrom.begin(), chrom.end(), rng_);
            if (random_real() < init_two_opt_prob_) chrom = two_opt(chrom, dist_matrix_, two_opt_passes_init_);
            pop.push_back({chrom, total_distance(chrom, dist_matrix_)});
        }
        return pop;
    }

    void generate_offspring_batch(const std::vector<Individual>& global_parents, int num_offspring_needed, std::vector<Individual>& offspring_batch) {
        // [修改 1] 這裡計算出來的現在是 CDF (累積機率)
        const std::vector<double> cdf = compute_cdf(global_parents);
        
        offspring_batch.clear();
        
        const auto best_parent_it = std::min_element(
            global_parents.begin(), global_parents.end(),
            [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
        if (best_parent_it != global_parents.end()) {
             offspring_batch.push_back(*best_parent_it);
        }

        while (static_cast<int>(offspring_batch.size()) < num_offspring_needed) {
            // [修改 2] 傳入 CDF 進行二分搜尋
            const Individual& p1 = roulette_wheel(global_parents, cdf);
            const Individual& p2 = roulette_wheel(global_parents, cdf);

            if (random_real() < crossover_rate_) {
                auto children = crossover(p1, p2);
                append_child(offspring_batch, children.first, num_offspring_needed);
                if (static_cast<int>(offspring_batch.size()) < num_offspring_needed)
                    append_child(offspring_batch, children.second, num_offspring_needed);
            } else {
                append_child(offspring_batch, p1, num_offspring_needed);
                if (static_cast<int>(offspring_batch.size()) < num_offspring_needed)
                    append_child(offspring_batch, p2, num_offspring_needed);
            }
        }
    }

    const DistanceMatrix& get_dist_matrix() const { return dist_matrix_; }

private:
    std::vector<City> cities_;
    DistanceMatrix dist_matrix_;
    double crossover_rate_, mutation_rate_, init_two_opt_prob_, offspring_two_opt_prob_;
    int two_opt_passes_init_, two_opt_passes_offspring_;
    std::mt19937 rng_;
    
    double random_real() {
        static std::uniform_real_distribution<double> dist(0.0, 1.0); 
        return dist(rng_);
    }
    int random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }

    // [修改] 計算 CDF (累積機率分佈)
    // 原本叫 fitness_prob，這裡為了語意清楚改名 compute_cdf，或保留原名也可
    std::vector<double> compute_cdf(const std::vector<Individual>& pop) {
        std::vector<double> cdf;
        cdf.reserve(pop.size());
        
        double total_inverse_fitness = 0.0;
        // 1. 先算總分母
        for (const auto& ind : pop) {
            total_inverse_fitness += 1.0 / ind.distance;
        }

        // 2. 再算累積機率
        double accumulated_prob = 0.0;
        for (const auto& ind : pop) {
            double prob = (1.0 / ind.distance) / total_inverse_fitness;
            accumulated_prob += prob;
            cdf.push_back(accumulated_prob);
        }
        
        // 強制最後一個為 1.0，避免浮點數誤差導致 lower_bound 找不到
        if (!cdf.empty()) cdf.back() = 1.0;
        
        return cdf;
    }

    // [修改] 使用 std::lower_bound (二分搜尋)
    const Individual& roulette_wheel(const std::vector<Individual>& pop, const std::vector<double>& cdf) {
        const double r = random_real();
        // lower_bound 回傳第一個 >= r 的位置，時間複雜度 O(log N)
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        
        size_t idx = std::distance(cdf.begin(), it);
        if (idx >= pop.size()) idx = pop.size() - 1;
        
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
                child[i] = a[i]; used[a[i]] = true;
            }
            int idx_b = (cut2 + 1) % n;
            int idx_child = (cut2 + 1) % n;
            for (int i = 0; i < n; ++i) {
                int cand = b[idx_b];
                if (!used[cand]) {
                    child[idx_child] = cand; used[cand] = true;
                    idx_child = (idx_child + 1) % n;
                }
                idx_b = (idx_b + 1) % n;
            }
            return child;
        };
        return {{create_child(p1.chromosome, p2.chromosome), 0.0}, 
                {create_child(p2.chromosome, p1.chromosome), 0.0}}; 
    }

    void mutate(std::vector<int>& chrom) {
        if (random_real() < mutation_rate_) {
            int n = chrom.size();
            std::swap(chrom[random_int(0, n - 1)], chrom[random_int(0, n - 1)]);
        }
    }

    void append_child(std::vector<Individual>& pop, Individual child, int limit) {
        if (static_cast<int>(pop.size()) >= limit) return;
        mutate(child.chromosome);
        if (random_real() < offspring_two_opt_prob_) 
            child.chromosome = two_opt(child.chromosome, dist_matrix_, two_opt_passes_offspring_);
        child.distance = total_distance(child.chromosome, dist_matrix_);
        pop.push_back(std::move(child));
    }
};

// --- Main ---

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start_time = MPI_Wtime();

    const std::string dataset = (argc > 1) ? argv[1] : "qa194.tsp";
    const int total_population = 8192;
    const int local_offspring_count = total_population / size;
    const int n_generations = 3000;
    
    // GA Parameters
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

        std::vector<int> global_pop_flat(total_population * num_cities);
        std::vector<double> global_fitness_flat(total_population); 

        std::vector<int> local_offspring_flat(local_offspring_count * num_cities);
        std::vector<double> local_fitness_flat(local_offspring_count); 

        std::vector<Individual> global_objs; 
        global_objs.reserve(total_population);
        std::vector<Individual> local_new_objs; 
        local_new_objs.reserve(local_offspring_count);

        if (rank == 0) {
            std::cout << "Starting MPI GA (Binary Search Selection) for TSP (" << dataset << ")..." << std::endl;
            std::vector<Individual> init_pop = worker.create_initial_population(total_population);
            flatten_population(init_pop, global_pop_flat, global_fitness_flat);
        }

        double total_comm_time = 0.0;
        double comm_start, comm_end;

        MPI_Barrier(MPI_COMM_WORLD); 
        double parallel_start_time = MPI_Wtime();

        for (int gen = 0; gen < n_generations; ++gen) {
            
            comm_start = MPI_Wtime();
            MPI_Bcast(global_pop_flat.data(), total_population * num_cities, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(global_fitness_flat.data(), total_population, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            comm_end = MPI_Wtime();
            total_comm_time += (comm_end - comm_start);
            
            reconstruct_population(global_objs, global_pop_flat, global_fitness_flat, num_cities);

            Individual best_parent;
            if (rank == 0) {
                 auto it = std::min_element(global_objs.begin(), global_objs.end(), 
                     [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
                 best_parent = *it;
            }

            // --- Computation ---
            worker.generate_offspring_batch(global_objs, local_offspring_count, local_new_objs);
            
            flatten_population(local_new_objs, local_offspring_flat, local_fitness_flat);

            comm_start = MPI_Wtime();
            MPI_Gather(local_offspring_flat.data(), local_offspring_count * num_cities, MPI_INT,
                       global_pop_flat.data(), local_offspring_count * num_cities, MPI_INT,
                       0, MPI_COMM_WORLD);
            MPI_Gather(local_fitness_flat.data(), local_offspring_count, MPI_DOUBLE,
                       global_fitness_flat.data(), local_offspring_count, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);
            comm_end = MPI_Wtime();
            total_comm_time += (comm_end - comm_start);

            if (rank == 0) {
                 std::copy(best_parent.chromosome.begin(), best_parent.chromosome.end(), global_pop_flat.begin());
                 global_fitness_flat[0] = best_parent.distance; 

                 if (gen % 100 == 0) std::cout << "Generation " << gen << " Best: " << best_parent.distance << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double parallel_end_time = MPI_Wtime();
        double total_end_time = MPI_Wtime();
        
        if (rank == 0) {
            double total_time = total_end_time - total_start_time;
            double parallel_time = parallel_end_time - parallel_start_time;
            double serial_time = total_time - parallel_time;
            double computation_time = parallel_time - total_comm_time;
            double comm_percent = (total_comm_time / parallel_time) * 100.0;
            double comp_percent = (computation_time / parallel_time) * 100.0;

            reconstruct_population(global_objs, global_pop_flat, global_fitness_flat, num_cities);
            auto best_it = std::min_element(global_objs.begin(), global_objs.end(),
                [](const Individual& a, const Individual& b) { return a.distance < b.distance; });

            std::cout << "\n=== Optimized Performance  ===" << std::endl;
            std::cout << "Total Time:       " << total_time << " s" << std::endl;
            std::cout << "Parallel Time:    " << parallel_time << " s" << std::endl;
            std::cout << "  - Comp Time:    " << computation_time << " s (" << comp_percent << "%)" << std::endl;
            std::cout << "  - Comm Time:    " << total_comm_time << " s (" << comm_percent << "%)" << std::endl;
            std::cout << "Final Best:       " << best_it->distance << std::endl;
        }

    } catch (const std::exception& ex) {
        std::cerr << "Rank " << rank << " Error: " << ex.what() << std::endl;
        MPI_Finalize(); return 1;
    }
    MPI_Finalize(); return 0;
}