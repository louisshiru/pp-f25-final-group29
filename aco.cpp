#include "lib/aco.h"
#include "lib/dataloader.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>

ACO::ACO(const std::vector<City>& cities, int num_ants, int max_iterations, double alpha, double beta, double evaporation, double q)
    : cities(cities), num_cities(cities.size()), num_ants(num_ants), max_iterations(max_iterations),
      alpha(alpha), beta(beta), evaporation(evaporation), Q(q) {
    
    std::random_device rd;
    rng = std::mt19937(rd());
    initialize();
}

double ACO::calculate_distance(const City& c1, const City& c2) {
    return std::sqrt(std::pow(c1.x - c2.x, 2) + std::pow(c1.y - c2.y, 2));
}

void ACO::initialize() {
    dist_matrix.resize(num_cities, std::vector<double>(num_cities));
    pheromone_matrix.resize(num_cities, std::vector<double>(num_cities));

    for (int i = 0; i < num_cities; ++i) {
        for (int j = 0; j < num_cities; ++j) {
            if (i != j) {
                dist_matrix[i][j] = calculate_distance(cities[i], cities[j]);
            } else {
                dist_matrix[i][j] = 0.0;
            }
            pheromone_matrix[i][j] = 1.0; // Initial pheromone level
        }
    }
}

int ACO::select_next_city(int current_city, const std::vector<bool>& visited) {
    std::vector<double> probabilities(num_cities, 0.0);
    double sum_probabilities = 0.0;

    for (int i = 0; i < num_cities; ++i) {
        if (!visited[i]) {
            double eta = 1.0 / dist_matrix[current_city][i];
            double tau = pheromone_matrix[current_city][i];
            probabilities[i] = std::pow(tau, alpha) * std::pow(eta, beta);
            sum_probabilities += probabilities[i];
        }
    }

    if (sum_probabilities == 0.0) {
        // Should not happen unless graph is disconnected or all visited
        for (int i = 0; i < num_cities; ++i) {
            if (!visited[i]) return i;
        }
        return -1;
    }

    std::uniform_real_distribution<> dis(0.0, sum_probabilities);
    double r = dis(rng);
    double current_sum = 0.0;

    for (int i = 0; i < num_cities; ++i) {
        if (!visited[i]) {
            current_sum += probabilities[i];
            if (current_sum >= r) {
                return i;
            }
        }
    }
    
    // Fallback (rounding errors)
    for (int i = num_cities - 1; i >= 0; --i) {
        if (!visited[i]) return i;
    }
    return -1;
}

void ACO::construct_solutions(std::vector<Ant>& ants) {
    for (int k = 0; k < num_ants; ++k) {
        ants[k].tour.clear();
        ants[k].visited.assign(num_cities, false);
        ants[k].tour_length = 0.0;

        // Start from a random city
        std::uniform_int_distribution<> dis(0, num_cities - 1);
        int start_city = dis(rng);
        
        ants[k].tour.push_back(start_city);
        ants[k].visited[start_city] = true;

        int current_city = start_city;
        for (int i = 1; i < num_cities; ++i) {
            int next_city = select_next_city(current_city, ants[k].visited);
            ants[k].tour.push_back(next_city);
            ants[k].visited[next_city] = true;
            ants[k].tour_length += dist_matrix[current_city][next_city];
            current_city = next_city;
        }
        // Return to start
        ants[k].tour_length += dist_matrix[current_city][start_city];
    }
}

void ACO::update_pheromones(const std::vector<Ant>& ants) {
    // Evaporation
    for (int i = 0; i < num_cities; ++i) {
        for (int j = 0; j < num_cities; ++j) {
            pheromone_matrix[i][j] *= (1.0 - evaporation);
            // Lower bound for pheromone to prevent stagnation
             if (pheromone_matrix[i][j] < 0.0001) pheromone_matrix[i][j] = 0.0001;
        }
    }

    // Deposit
    for (const auto& ant : ants) {
        double deposit = Q / ant.tour_length;
        for (size_t i = 0; i < ant.tour.size(); ++i) {
            int from = ant.tour[i];
            int to = ant.tour[(i + 1) % ant.tour.size()];
            pheromone_matrix[from][to] += deposit;
            pheromone_matrix[to][from] += deposit; // Symmetric TSP
        }
    }
}

void ACO::run() {
    double best_global_length = std::numeric_limits<double>::max();
    std::vector<int> best_global_tour;

    std::vector<Ant> ants(num_ants);

    for (int iter = 0; iter < max_iterations; ++iter) {
        construct_solutions(ants);
        update_pheromones(ants);

        double best_iter_length = std::numeric_limits<double>::max();
        int best_ant_idx = -1;

        for (int k = 0; k < num_ants; ++k) {
            if (ants[k].tour_length < best_iter_length) {
                best_iter_length = ants[k].tour_length;
                best_ant_idx = k;
            }
        }

        if (best_iter_length < best_global_length) {
            best_global_length = best_iter_length;
            best_global_tour = ants[best_ant_idx].tour;
            std::cout << "Iteration " << iter << ": New best length = " << best_global_length << std::endl;
        }
    }

    std::cout << "Final Best Distance: " << best_global_length << std::endl;
    std::cout << "Best Route: ";
    for (int city_id : best_global_tour) {
        std::cout << city_id << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    const std::string dataset = (argc > 1) ? argv[1] : "za929.tsp";

    // ACO Parameters
    int num_ants = 30; // Can be tuned
    int max_iterations = 100;
    double alpha = 1.0;
    double beta = 2.0;
    double evaporation = 0.5;
    double Q = 100.0;

    try {
        Dataloader dl(dataset);
        
        // Adjust num_ants based on problem size if needed
        // num_ants = dl.cities.size(); 

        std::cout << "Starting ACO for TSP (" << dataset << ")..." << std::endl;
        std::cout << "Ants: " << num_ants << ", Iterations: " << max_iterations << std::endl;
        std::cout << "Alpha: " << alpha << ", Beta: " << beta << ", Evaporation: " << evaporation << ", Q: " << Q << std::endl;

        ACO aco(dl.cities, num_ants, max_iterations, alpha, beta, evaporation, Q);

        auto start = std::chrono::high_resolution_clock::now();
        aco.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time: " << elapsed.count() << " s" << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Failed to load dataset '" << dataset << "': " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
