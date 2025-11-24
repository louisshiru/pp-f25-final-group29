#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>
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

int main() {
    Dataloader dl;
    const auto& cities = dl.cities;
    int n = cities.size();

    if (n == 0) return 0;

    // We fix the first city to 0 to avoid checking rotated versions of the same tour.
    // We permute the remaining cities indices [1, 2, ..., n-1].
    std::vector<int> p;
    for (int i = 1; i < n; ++i) {
        p.push_back(i);
    }

    double min_dist = std::numeric_limits<double>::max();
    std::vector<int> best_route;

    // Initial check for the sorted permutation
    do {
        std::vector<int> current_route;
        current_route.push_back(0); // Start with city 0
        current_route.insert(current_route.end(), p.begin(), p.end());

        double d = calculate_total_distance(current_route, cities);
        if (d < min_dist) {
            min_dist = d;
            best_route = current_route;
        }

    } while (std::next_permutation(p.begin(), p.end()));

    std::cout << "Brute Force TSP Result:" << std::endl;
    std::cout << "Minimum Distance: " << std::fixed << std::setprecision(4) << min_dist << std::endl;
    std::cout << "Best Route: ";
    for (int city_idx : best_route) {
        std::cout << city_idx << " ";
    }
    std::cout << std::endl;

    return 0;
}
