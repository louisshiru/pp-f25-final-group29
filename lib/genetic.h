#ifndef GENETIC_H
#define GENETIC_H

#include <vector>
#include <utility>
#include "dataloader.h"

struct Individual {
    std::vector<int> chromosome;
    double fitness;
};

class GeneticAlgorithm {
public:
    GeneticAlgorithm(const std::vector<City>& cities, int n_population, int n_generations, double crossover_per, double mutation_per);
    void run();

private:
    std::vector<City> cities;
    int n_population;
    int n_generations;
    double crossover_per;
    double mutation_per;
    std::vector<Individual> population;

    std::vector<Individual> initial_population();
    double dist_two_cities(const City& c1, const City& c2);
    double total_dist_individual(const std::vector<int>& chromosome);
    std::vector<double> fitness_prob(const std::vector<Individual>& pop);
    Individual roulette_wheel(const std::vector<Individual>& pop, const std::vector<double>& probs);
    std::pair<Individual, Individual> crossover(const Individual& p1, const Individual& p2);
    Individual mutation(Individual ind);
    
    // Helper for random numbers
    double random_0_1();
    int random_int(int min, int max);
};

#endif
