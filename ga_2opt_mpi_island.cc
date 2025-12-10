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

    std::vector<Individual> create_initial_population(int pop_size) {
        std::vector<Individual> pop;
        std::vector<int> base(cities_.size());
        std::iota(base.begin(), base.end(), 0);

        pop.reserve(pop_size);
        for (int i = 0; i < pop_size; ++i) {
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

    std::vector<Individual> generate_offspring_batch(const std::vector<Individual>& parents, int num_offspring_needed) {
        const std::vector<double> probs = fitness_prob(parents);
        std::vector<Individual> offspring_batch;
        offspring_batch.reserve(num_offspring_needed);
        
        while (static_cast<int>(offspring_batch.size()) < num_offspring_needed) {
            const Individual p1 = roulette_wheel(parents, probs);
            const Individual p2 = roulette_wheel(parents, probs);

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
        if (random_real() < offspring_two_opt_prob_) {
            child.chromosome = two_opt(child.chromosome, dist_matrix_, two_opt_passes_offspring_);
        }
        child.distance = total_distance(child.chromosome, dist_matrix_);
        pop.push_back(std::move(child));
    }
};

void migrate_elites(std::vector<Individual>& pop, int best_num, int rank, int size, int num_cities, const DistanceMatrix& dist) {
    std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
        return a.distance < b.distance;
    });

    std::vector<int> send_buf;
    send_buf.reserve(best_num * num_cities);
    for (int i = 0; i < best_num; ++i) {
        send_buf.insert(send_buf.end(), pop[i].chromosome.begin(), pop[i].chromosome.end());
    }

    std::vector<int> recv_buf(best_num * num_cities);

    int next_rank = (rank + 1) % size;
    int prev_rank = (rank - 1 + size) % size;

    MPI_Request reqs[2];
    MPI_Status stats[2];

    MPI_Isend(send_buf.data(), best_num * num_cities, MPI_INT, next_rank, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(recv_buf.data(), best_num * num_cities, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Waitall(2, reqs, stats);

    // 將外來菁英加入本地 (取代最差的個體)
    // 因為我們已經排序過 pop，最差的在最後面
    for (int i = 0; i < best_num; ++i) {
        std::vector<int> chrom(recv_buf.begin() + i * num_cities, recv_buf.begin() + (i + 1) * num_cities);
        double d = total_distance(chrom, dist);
        
        // 替換倒數第 i 個 (從最後面開始)
        int replace_idx = pop.size() - 1 - i;
        if (replace_idx >= 0) {
            pop[replace_idx] = {chrom, d};
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string dataset = (argc > 1) ? argv[1] : "qa194.tsp";
    
    const int total_population_target = 8192;
    const int local_pop_size = total_population_target / size; 

    const int n_generations = 3000;
    
    // Island Model 遷移參數
    const int migration_interval = 200; // 每 200 代交換一次
    const int migration_count = 50;     // 每次交換 50 個菁英

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

        if (rank == 0) {
            std::cout << "Starting MPI GA (Island Model) for TSP (" << dataset << ")..." << std::endl;
            std::cout << "Nodes: " << size << ", Local Pop: " << local_pop_size << ", Total Pop approx: " << local_pop_size * size << std::endl;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // 本地初始化 (每個 Rank 自己做，不廣播)
        std::vector<Individual> population = worker.create_initial_population(local_pop_size);

        // --- 演化迴圈 ---
        for (int gen = 0; gen < n_generations; ++gen) {
            
            // A. 找出本地最佳 (為了 Elitism 和觀測)
            auto best_it = std::min_element(
                population.begin(), population.end(),
                [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
            Individual best_local = *best_it;

            // B. 產生下一代 (使用本地族群當父母)
            std::vector<Individual> offspring = worker.generate_offspring_batch(population, local_pop_size);
            
            // C. Elitism: 強制保留上一代本地最強
            offspring[0] = best_local;
            
            // 更新族群
            population = std::move(offspring);

            // D. 遷移 (Migration)
            if (gen > 0 && gen % migration_interval == 0) {
                migrate_elites(population, migration_count, rank, size, num_cities, worker.get_dist_matrix());
            }

            // E. 進度顯示 (由 Rank 0 代表顯示)
            if (rank == 0 && (gen % 100 == 0)) {
                auto current_best = std::min_element(
                    population.begin(), population.end(),
                    [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
                std::cout << "Generation " << gen << " (Rank 0) Best: " << current_best->distance << std::endl;
            }
        }

        // 找出本地最終最佳解
        auto final_local_best_it = std::min_element(
            population.begin(), population.end(),
            [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
        
        struct {
            double value;
            int rank;
        } local_res, global_res;

        local_res.value = final_local_best_it->distance;
        local_res.rank = rank;

        // 使用 MPI_Allreduce 找出全域最小值是誰
        MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();

        // 印出結果
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Time: " << elapsed.count() << " s" << std::endl;
            std::cout << "Final Best Distance: " << global_res.value << std::endl;
            std::cout << "Best solution found by Rank " << global_res.rank << std::endl;
        }

    } catch (const std::exception& ex) {
        std::cerr << "Rank " << rank << " Error: " << ex.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}