#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <numeric>
#include <random>
#include <chrono>
#include "dataloader.h"

using namespace std;

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

// Helper to reverse a segment in the tour
// Reverses the segment from index i to index j (inclusive)
void reverse_segment(std::vector<int>& route, int i, int j) {
    int n = route.size();
    int len = (j - i + n) % n + 1;
    int l = i;
    int r = j;
    for (int k = 0; k < len / 2; ++k) {
        std::swap(route[l], route[r]);
        l = (l + 1) % n;
        r = (r - 1 + n) % n;
    }
}

class Solver {
    const std::vector<City>& cities;
    int n;
    std::vector<std::vector<int>> adj;
    std::mt19937 rng;

public:
    Solver(const std::vector<City>& c) : cities(c), n(c.size()) {
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        compute_candidates();
    }

    void compute_candidates() {
        adj.resize(n);
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<double, int>> neighbors;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                neighbors.push_back({dist(cities[i], cities[j]), j});
            }
            std::sort(neighbors.begin(), neighbors.end());
            int limit = std::min((int)neighbors.size(), 20);
            for (int k = 0; k < limit; ++k) {
                adj[i].push_back(neighbors[k].second);
            }
        }
    }

    // 2-opt with candidate list
    void local_search(std::vector<int>& route) {
        bool improved = true;
        std::vector<int> pos(n);
        
        while (improved) {
            improved = false;
            for(int i=0; i<n; ++i) pos[route[i]] = i; // Update positions

            for (int i = 0; i < n; ++i) {
                int t1 = route[i];
                int t2_idx = (i + 1) % n;
                int t2 = route[t2_idx];
                double base_dist = dist(cities[t1], cities[t2]);

                for (int t3 : adj[t2]) {
                    if (t3 == t1) continue;
                    
                    // We want to add edge (t2, t3).
                    // This implies breaking (t1, t2) and some edge connected to t3.
                    // To form a valid tour by adding (t2, t3), we must reverse the segment between t3 and t1.
                    // The edge broken at t3 will be (t3_prev, t3).
                    
                    int t3_idx = pos[t3];
                    int t3_prev_idx = (t3_idx - 1 + n) % n;
                    int t3_prev = route[t3_prev_idx];

                    double current_len = base_dist + dist(cities[t3_prev], cities[t3]);
                    double new_len = dist(cities[t2], cities[t3]) + dist(cities[t1], cities[t3_prev]);
                    
                    if (new_len < current_len - 1e-9) {
                        reverse_segment(route, t3_idx, i); // Reverse from t3 to t1
                        improved = true;
                        goto next_iter;
                    }
                }
            }
            next_iter:;
        }
    }

    void double_bridge(std::vector<int>& route) {
        if (n < 8) return;
        std::vector<int> p(4);
        for(int& x : p) x = rng() % n;
        std::sort(p.begin(), p.end());
        // Ensure some separation
        if (p[0] + 1 >= p[1] || p[1] + 1 >= p[2] || p[2] + 1 >= p[3]) return; 
        
        // Segments: A=[0, p0], B=[p0+1, p1], C=[p1+1, p2], D=[p2+1, p3], E=[p3+1, n-1]
        // New tour: A D C B E
        
        std::vector<int> new_route;
        new_route.reserve(n);
        
        auto append = [&](int start, int end) {
            for (int i = start; i <= end; ++i) new_route.push_back(route[i]);
        };
        
        append(0, p[0]);             // A
        append(p[2] + 1, p[3]);      // D
        append(p[1] + 1, p[2]);      // C
        append(p[0] + 1, p[1]);      // B
        append(p[3] + 1, n - 1);     // E
        
        route = new_route;
    }

    std::vector<int> solve() {
        std::vector<int> current_route(n);
        std::iota(current_route.begin(), current_route.end(), 0);
        std::shuffle(current_route.begin(), current_route.end(), rng);
        
        local_search(current_route);
        
        std::vector<int> best_route = current_route;
        double best_dist = calculate_total_distance(best_route, cities);
        
        // Iterated Local Search
        int max_iter = 100; 
        if (n > 1000) max_iter = 50;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<int> candidate = best_route;
            double_bridge(candidate);
            local_search(candidate);
            
            double dist = calculate_total_distance(candidate, cities);
            if (dist < best_dist) {
                best_dist = dist;
                best_route = candidate;
                iter = 0; // Reset counter if improvement found
            }
        }
        
        return best_route;
    }
};

int main(int argc, char** argv) {
    const std::string dataset = (argc > 1) ? argv[1] : "za929.tsp";

    try {
        Dataloader dl(dataset);
        const auto& cities = dl.cities;
        int n = cities.size();

        if (n == 0) return 0;

        Solver solver(cities);

        auto start = std::chrono::high_resolution_clock::now();
        const auto optimized_route = solver.solve();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        const double best_distance = calculate_total_distance(optimized_route, cities);

        std::cout << "LKH TSP Result (" << dataset << "):" << std::endl;
        std::cout << "Time: " << elapsed.count() << " s" << std::endl;
        std::cout << "Distance: " << std::fixed << std::setprecision(4) << best_distance << std::endl;
        std::cout << "Route: ";
        for (size_t i = 0; i < optimized_route.size(); ++i) {
            std::cout << optimized_route[i] << (i == optimized_route.size() - 1 ? "" : " ");
        }
        std::cout << std::endl; 
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load dataset '" << dataset << "': " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
