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

// --- GA Worker Class (已修改 Selection 邏輯) ---
class GAWorker {
public:
    GAWorker(const std::vector<City>& cities, double crossover_rate, double mutation_rate, 
             double init_two_opt_prob, double offspring_two_opt_prob, int two_opt_passes_init, int two_opt_passes_offspring, int rank)
        : cities_(cities), dist_matrix_(cities), crossover_rate_(crossover_rate), mutation_rate_(mutation_rate),
          init_two_opt_prob_(init_two_opt_prob), offspring_two_opt_prob_(offspring_two_opt_prob),
          two_opt_passes_init_(two_opt_passes_init), two_opt_passes_offspring_(two_opt_passes_offspring), rng_(rank + 1) {}

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

    // 修改：使用 compute_cdf 和 新的 roulette_wheel
    std::vector<Individual> generate_offspring_batch(const std::vector<Individual>& parents, int num_offspring_needed) {
        // [Change 1] 計算累積機率 (CDF)
        const std::vector<double> cdf = compute_cdf(parents);
        
        std::vector<Individual> offspring_batch;
        offspring_batch.reserve(num_offspring_needed);
        
        while (static_cast<int>(offspring_batch.size()) < num_offspring_needed) {
            // [Change 2] 傳入 CDF 進行二分搜尋
            const Individual& p1 = roulette_wheel(parents, cdf);
            const Individual& p2 = roulette_wheel(parents, cdf);

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

    // [New] 計算累積機率分佈 (CDF)
    std::vector<double> compute_cdf(const std::vector<Individual>& pop) {
        std::vector<double> cdf;
        cdf.reserve(pop.size());
        
        double total_inverse_fitness = 0.0;
        for (const auto& ind : pop) {
            total_inverse_fitness += 1.0 / ind.distance;
        }

        double accumulated_prob = 0.0;
        for (const auto& ind : pop) {
            double prob = (1.0 / ind.distance) / total_inverse_fitness;
            accumulated_prob += prob;
            cdf.push_back(accumulated_prob);
        }
        if (!cdf.empty()) cdf.back() = 1.0;
        return cdf;
    }

    // [New] 使用 std::lower_bound 進行二分搜尋
    const Individual& roulette_wheel(const std::vector<Individual>& pop, const std::vector<double>& cdf) {
        const double r = random_real();
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
        if (random_real() < offspring_two_opt_prob_) {
            child.chromosome = two_opt(child.chromosome, dist_matrix_, two_opt_passes_offspring_);
        }
        child.distance = total_distance(child.chromosome, dist_matrix_);
        pop.push_back(std::move(child));
    }
};

// --- Migration Function (包含計時邏輯) ---
double migrate_elites(std::vector<Individual>& pop, int best_num, int rank, int size, int num_cities, const DistanceMatrix& dist) {
    // 排序算 computation，先不計入通訊時間
    std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
        return a.distance < b.distance;
    });
    
    // 開始通訊計時
    double mpi_start = MPI_Wtime();

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

    double mpi_end = MPI_Wtime();

    // 替換最差個體 (算 computation)
    for (int i = 0; i < best_num; ++i) {
        std::vector<int> chrom(recv_buf.begin() + i * num_cities, recv_buf.begin() + (i + 1) * num_cities);
        double d = total_distance(chrom, dist);
        int replace_idx = pop.size() - 1 - i;
        if (replace_idx >= 0) {
            pop[replace_idx] = {chrom, d};
        }
    }
    
    return mpi_end - mpi_start;
}

// --- Main ---

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start_time = MPI_Wtime();

    // const std::string dataset = (argc > 1) ? argv[1] : "qa194.tsp";
    // const int total_population_target = 8192;
    // const int local_pop_size = total_population_target / size; 
    // const int n_generations = 3000;
    // const double crossover_rate = 0.8;
    // const double mutation_rate = 0.2;
    // const double init_two_opt_prob = 1.0;         
    // const double offspring_two_opt_prob = 1.0;    
    // const int two_opt_passes_init = 100;          
    // const int two_opt_passes_offspring = 2;     

    const std::string dataset = (argc > 1) ? argv[1] : "zi929.tsp";
    const int total_population_target = 8192;
    const int local_pop_size = total_population_target / size; 
    const int n_generations = 8000;
    const double crossover_rate = 0.8;
    const double mutation_rate = 0.2;
    const double init_two_opt_prob = 1.0;
    const double offspring_two_opt_prob = 1.0;
    const int two_opt_passes_init = 100;
    const int two_opt_passes_offspring = 2;

    const int migration_interval = 200; 
    const int migration_count = 50; 

    try {
        Dataloader dl(dataset);
        int num_cities = dl.cities.size();

        GAWorker worker(dl.cities, crossover_rate, mutation_rate, 
                         init_two_opt_prob, offspring_two_opt_prob, two_opt_passes_init, two_opt_passes_offspring, rank);

        if (rank == 0) {
            std::cout << "Starting MPI GA (Island Model + Binary Search) for TSP (" << dataset << ")..." << std::endl;
            std::cout << "Nodes: " << size << ", Local Pop: " << local_pop_size << ", Total Pop approx: " << local_pop_size * size << std::endl;
        }

        double total_comm_time = 0.0;
        
        MPI_Barrier(MPI_COMM_WORLD);
        double parallel_start_time = MPI_Wtime();

        std::vector<Individual> population = worker.create_initial_population(local_pop_size);

        for (int gen = 0; gen < n_generations; ++gen) {
            
            // A. 本地運算
            auto best_it = std::min_element(
                population.begin(), population.end(),
                [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
            Individual best_local = *best_it;

            std::vector<Individual> offspring = worker.generate_offspring_batch(population, local_pop_size);
            
            offspring[0] = best_local; 
            population = std::move(offspring);

            // B. 遷移
            if (gen > 0 && gen % migration_interval == 0) {
                double comm_elapsed = migrate_elites(population, migration_count, rank, size, num_cities, worker.get_dist_matrix());
                total_comm_time += comm_elapsed;
            }

            // C. 進度顯示
            if (rank == 0 && (gen % 100 == 0)) {
                auto current_best = std::min_element(
                    population.begin(), population.end(),
                    [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
                std::cout << "Generation " << gen << " (Rank 0) Best: " << current_best->distance << std::endl;
            }
        }

        auto final_local_best_it = std::min_element(
            population.begin(), population.end(),
            [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
        
        struct {
            double value;
            int rank;
        } local_res, global_res;

        local_res.value = final_local_best_it->distance;
        local_res.rank = rank;

        // [Time] Final Communication (Allreduce)
        double reduce_start = MPI_Wtime();
        MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        double reduce_end = MPI_Wtime();
        total_comm_time += (reduce_end - reduce_start);

        // --- [修正開始] 測量最後的同步等待時間 ---
        double barrier_start = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD); 
        double barrier_end = MPI_Wtime();
        
        // 將這段等待時間也加入通訊/同步成本
        total_comm_time += (barrier_end - barrier_start);
        // --- [修正結束] ---

        double parallel_end_time = MPI_Wtime();
        double total_end_time = MPI_Wtime();

        if (rank == 0) {
            double total_time = total_end_time - total_start_time;
            double parallel_time = parallel_end_time - parallel_start_time;
            double serial_time = total_time - parallel_time; 
            double computation_time = parallel_time - total_comm_time;
            
            double comm_percent = (total_comm_time / parallel_time) * 100.0;
            double comp_percent = (computation_time / parallel_time) * 100.0;

            std::cout << "\n=== Island Model Performance ===" << std::endl;
            std::cout << "Total Execution Time:       " << total_time << " s" << std::endl;
            std::cout << "Serial (Init/IO) Time:      " << serial_time << " s" << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
            std::cout << "Parallel Region Time:       " << parallel_time << " s (100%)" << std::endl;
            std::cout << "  - Computation Time:       " << computation_time << " s (" << comp_percent << "%)" << std::endl;
            std::cout << "  - Communication Time:     " << total_comm_time << " s (" << comm_percent << "%)" << std::endl;
            std::cout << "============================================" << std::endl;

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