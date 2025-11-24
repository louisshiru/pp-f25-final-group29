#include "dataloader.h"
#include <vector>

Dataloader::Dataloader() {
    std::vector<double> x = {0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5};
    std::vector<double> y = {1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12};
    
    for (int i = 0; i < 10; ++i) {
        cities.push_back({i, x[i], y[i]});
    }

    // [TODO]: load a TSP dataset from file
    // sources:
    // 1. https://www.math.uwaterloo.ca/tsp/data/index.html
    // 2. https://www.kaggle.com/datasets/ziya07/traveling-salesman-problem-tsplib-dataset
}
