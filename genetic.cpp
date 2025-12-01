#include "genetic.h"
#include "dataloader.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <stdexcept>
#include <chrono>

GeneticAlgorithm::GeneticAlgorithm(const std::vector<City>& cities, int n_population, int n_generations, double crossover_per, double mutation_per)
    : cities(cities), n_population(n_population), n_generations(n_generations), crossover_per(crossover_per), mutation_per(mutation_per) {
    population = initial_population();
}

double GeneticAlgorithm::random_0_1() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}

int GeneticAlgorithm::random_int(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

std::vector<Individual> GeneticAlgorithm::initial_population() {
    std::vector<Individual> pop;
    std::vector<int> base_chromosome;
    for (int i = 0; i < cities.size(); ++i) {
        base_chromosome.push_back(i);
    }

    for (int i = 0; i < n_population; ++i) {
        std::vector<int> chrom = base_chromosome;
        std::shuffle(chrom.begin(), chrom.end(), std::mt19937(std::random_device()()));
        double dist = total_dist_individual(chrom);
        pop.push_back({chrom, dist});
    }
    return pop;
}

double GeneticAlgorithm::dist_two_cities(const City& c1, const City& c2) {
    return std::sqrt(std::pow(c1.x - c2.x, 2) + std::pow(c1.y - c2.y, 2));
}

double GeneticAlgorithm::total_dist_individual(const std::vector<int>& chromosome) {
    double total_dist = 0.0;
    for (size_t i = 0; i < chromosome.size(); ++i) {
        const City& c1 = cities[chromosome[i]];
        const City& c2 = cities[chromosome[(i + 1) % chromosome.size()]];
        total_dist += dist_two_cities(c1, c2);
    }
    return total_dist;
}

std::vector<double> GeneticAlgorithm::fitness_prob(const std::vector<Individual>& pop) {
    std::vector<double> fitness_vals;
    double total_fitness = 0.0;
    for (const auto& ind : pop) {
        double f = 1.0 / ind.fitness;
        fitness_vals.push_back(f);
        total_fitness += f;
    }
    
    std::vector<double> probs;
    for (double f : fitness_vals) {
        probs.push_back(f / total_fitness);
    }
    return probs;
}

Individual GeneticAlgorithm::roulette_wheel(const std::vector<Individual>& pop, const std::vector<double>& probs) {
    double r = random_0_1();
    double cum_prob = 0.0;
    for (size_t i = 0; i < pop.size(); ++i) {
        cum_prob += probs[i];
        if (r <= cum_prob) {
            return pop[i];
        }
    }
    return pop.back();
}

std::pair<Individual, Individual> GeneticAlgorithm::crossover(const Individual& p1, const Individual& p2) {
    int n = p1.chromosome.size();
    int cut1 = random_int(0, n - 1);
    int cut2 = random_int(0, n - 1);
    if (cut1 > cut2) std::swap(cut1, cut2);

    auto create_child = [&](const std::vector<int>& parent1, const std::vector<int>& parent2) {
        std::vector<int> child(n, -1);
        std::set<int> in_child;
        
        for (int i = cut1; i <= cut2; ++i) {
            child[i] = parent1[i];
            in_child.insert(parent1[i]);
        }

        int current_p2_idx = (cut2 + 1) % n;
        int current_child_idx = (cut2 + 1) % n;
        
        for (int i = 0; i < n; ++i) {
             int candidate = parent2[current_p2_idx];
             if (in_child.find(candidate) == in_child.end()) {
                 child[current_child_idx] = candidate;
                 current_child_idx = (current_child_idx + 1) % n;
             }
             current_p2_idx = (current_p2_idx + 1) % n;
        }
        return child;
    };

    std::vector<int> c1_chrom = create_child(p1.chromosome, p2.chromosome);
    std::vector<int> c2_chrom = create_child(p2.chromosome, p1.chromosome);

    return {{c1_chrom, total_dist_individual(c1_chrom)}, {c2_chrom, total_dist_individual(c2_chrom)}};
}

Individual GeneticAlgorithm::mutation(Individual ind) {
    if (random_0_1() < mutation_per) {
        int n = ind.chromosome.size();
        int idx1 = random_int(0, n - 1);
        int idx2 = random_int(0, n - 1);
        std::swap(ind.chromosome[idx1], ind.chromosome[idx2]);
        ind.fitness = total_dist_individual(ind.chromosome);
    }
    return ind;
}

void GeneticAlgorithm::run() {
    
    // [TODO]: optimize ga process
    for (int gen = 0; gen < n_generations; ++gen) {
        std::vector<double> probs = fitness_prob(population);
        std::vector<Individual> new_population;

        // Elitism: keep the best one
        auto best_it = std::min_element(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
        new_population.push_back(*best_it);

        while (new_population.size() < n_population) {
            Individual p1 = roulette_wheel(population, probs);
            Individual p2 = roulette_wheel(population, probs);
            
            if (random_0_1() < crossover_per) {
                std::pair<Individual, Individual> children = crossover(p1, p2);
                new_population.push_back(mutation(children.first));
                if (new_population.size() < n_population) {
                    new_population.push_back(mutation(children.second));
                }
            } else {
                new_population.push_back(mutation(p1));
                if (new_population.size() < n_population) {
                    new_population.push_back(mutation(p2));
                }
            }
        }
        population = new_population;
        
        double best_dist = new_population[0].fitness;
        for(const auto& ind : population) {
            if(ind.fitness < best_dist) best_dist = ind.fitness;
        }
        
        if (gen % 100 == 0 || gen == n_generations - 1) {
             std::cout << "Generation " << gen << " Best Distance: " << best_dist << std::endl;
        }
    }
    
    auto best_it = std::min_element(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
        return a.fitness < b.fitness;
    });
    
    std::cout << "Final Best Distance: " << best_it->fitness << std::endl;
    std::cout << "Best Route: ";
    for (int city_id : best_it->chromosome) {
        std::cout << city_id << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    const std::string dataset = (argc > 1) ? argv[1] : "za929.tsp";

    int n_population = 1000;
    int n_generations = 5000;
    double crossover_per = 0.8;
    double mutation_per = 0.2;

    try {
        Dataloader dl(dataset);

        std::cout << "Starting Genetic Algorithm for TSP (" << dataset << ")..." << std::endl;
        std::cout << "Population: " << n_population << ", Generations: " << n_generations << std::endl;
        std::cout << "Crossover: " << crossover_per << ", Mutation: " << mutation_per << std::endl;

        GeneticAlgorithm ga(dl.cities, n_population, n_generations, crossover_per, mutation_per);
        
        auto start = std::chrono::high_resolution_clock::now();
        ga.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time: " << elapsed.count() << " s" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load dataset '" << dataset << "': " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
