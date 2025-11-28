#ifndef ACO_H
#define ACO_H

#include "dataloader.h"
#include <vector>
#include <random>

struct Ant {
    std::vector<int> tour;
    std::vector<bool> visited;
    double tour_length;
};

class ACO {
public:
    ACO(const std::vector<City>& cities, int num_ants, int max_iterations, double alpha, double beta, double evaporation, double q);
    void run();

private:
    const std::vector<City>& cities;
    int num_cities;
    int num_ants;
    int max_iterations;
    double alpha;
    double beta;
    double evaporation;
    double Q;

    std::vector<std::vector<double>> dist_matrix;
    std::vector<std::vector<double>> pheromone_matrix;
    
    // Random number generation
    std::mt19937 rng;

    double calculate_distance(const City& c1, const City& c2);
    void initialize();
    void construct_solutions(std::vector<Ant>& ants);
    void update_pheromones(const std::vector<Ant>& ants);
    int select_next_city(int current_city, const std::vector<bool>& visited);
};

#endif
