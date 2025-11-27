#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>

struct City {
    int id;
    double x;
    double y;
};

class Dataloader {
public:
    std::vector<City> cities;
    explicit Dataloader(const std::string& dataset_name = "qa194.tsp");
};

#endif
