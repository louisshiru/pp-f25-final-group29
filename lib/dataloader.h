#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>

struct City {
    int id;
    double x;
    double y;
};

class Dataloader {
public:
    std::vector<City> cities;
    Dataloader();
};

#endif
