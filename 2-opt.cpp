#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <numeric>
#include <chrono>
#include "dataloader.h"

// Distance function
double dist(const City& c1, const City& c2) {
    return std::sqrt(std::pow(c1.x - c2.x, 2) + std::pow(c1.y - c2.y, 2));
}

// Total distance for a route
double calculate_total_distance(const std::vector<int>& route, const std::vector<City>& cities) {
    double total = 0.0;
    for (size_t i = 0; i < route.size(); ++i) {
        const City& c1 = cities[route[i]];
        const City& c2 = cities[route[(i + 1) % route.size()]];
        total += dist(c1, c2);
    }
    return total;
}

std::vector<int> two_opt(const std::vector<int>& initial_route, const std::vector<City>& cities) {
    std::vector<int> route = initial_route;
    const size_t n = route.size();

    if (n < 4) return route;

    bool improved = true;
    while (improved) {
        improved = false;
        for (size_t i = 1; i < n - 1 && !improved; ++i) {
            for (size_t k = i + 1; k < n && !improved; ++k) {
                const size_t next = (k + 1) % n;

                const double delta =
                    dist(cities[route[i - 1]], cities[route[k]]) +
                    dist(cities[route[i]], cities[route[next]]) -
                    dist(cities[route[i - 1]], cities[route[i]]) -
                    dist(cities[route[k]], cities[route[next]]);

                if (delta < -1e-9) {
                    std::reverse(route.begin() + static_cast<long>(i), route.begin() + static_cast<long>(k) + 1);
                    improved = true;
                }
            }
        }
    }

    return route;
}

int main(int argc, char** argv) {
    const std::string dataset = (argc > 1) ? argv[1] : "qa194.tsp";

    try {
        Dataloader dl(dataset);
        const auto& cities = dl.cities;
        int n = cities.size();

        if (n == 0) return 0;

        std::vector<int> route(n);
        std::iota(route.begin(), route.end(), 0);

        auto start = std::chrono::high_resolution_clock::now();
        const auto optimized_route = two_opt(route, cities);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        const double best_distance = calculate_total_distance(optimized_route, cities);

        std::cout << "2-opt TSP Result (" << dataset << "):" << std::endl;
        std::cout << "Time: " << elapsed.count() << " s" << std::endl;
        std::cout << "Distance: " << std::fixed << std::setprecision(4) << best_distance << std::endl;
        std::cout << "Route: ";
        for (int city_idx : optimized_route) {
            std::cout << city_idx << " ";
        }
        std::cout << optimized_route.front() << std::endl; // close the tour
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load dataset '" << dataset << "': " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
