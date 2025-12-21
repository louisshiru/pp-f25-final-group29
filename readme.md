# PP-f25 Final Project: TSP with Genetic Algorithm

## Description
This project aims to solve the Traveling Salesman Problem (TSP) using a Genetic Algorithm. The primary objective is to apply parallel programming techniques to accelerate the computation.

We provide two solvers:
1.  **Genetic Algorithm (`tsp_ga`)**: A heuristic approach to find an approximate solution efficiently.
2.  **Brute Force (`tsp_bf`)**: An exact method to find the optimal solution, used for verification.
3.  **GA + 2-opt (`tsp_ga2opt`)**: GA with local 2-opt refinement.
4.  **GA + 2-opt (OpenMP) (`tsp_ga2opt_omp`)**: Parallelized GA + 2-opt using OpenMP.

## Usage

### Build All
Generate all executable objects:
```bash
make
```

### Run Genetic Algorithm
Run the genetic algorithm approach (Approximate Answer):
```bash
make tsp_ga && ./tsp_ga # Default use qa194.tsp (Qatar - 194 cities)
make tsp_ga && ./tsp_ga mu1979.tsp # Select mu1979.tsp  (Oman - 1,979 Cities)
```

### Run Genetic Algorithm + 2opt
Run the genetic algorithm approach with local search:
```bash
make tsp_ga2opt && ./tsp_ga2opt # Default use qa194.tsp (Qatar - 194 cities)
make tsp_ga2opt && ./tsp_ga2opt mu1979.tsp # Select mu1979.tsp  (Oman - 1,979 Cities)
```

### Run Genetic Algorithm + 2opt (OpenMP)
Parallel version (requires OpenMP-capable compiler):
```bash
make tsp_ga2opt_omp && ./tsp_ga2opt_omp # Default use qa194.tsp (Qatar - 194 cities)
make tsp_ga2opt_omp && ./tsp_ga2opt_omp mu1979.tsp # Select mu1979.tsp  (Oman - 1,979 Cities)
```

### Run Genetic Algorithm + 2opt (MPI)
```bash
make clean && make tsp_ga2opt_mpi && run --mpi=pmix -N 1 -n 12 -- ./tsp_ga2opt_mpi # Default use qa194.tsp (Qatar - 194 cities)
make clean && make tsp_ga2opt_mpi && run --mpi=pmix -N 1 -n 12 -- ./tsp_ga2opt_mpi mu1979.tsp # Select mu1979.tsp  (Oman - 1,979 Cities)
make make tsp_ga2opt_mpi_island && run --mpi=pmix -N 8 -n 8 -- ./tsp_ga2opt_mpi_island # Default use qa194.tsp (Qatar - 194 cities)
make make tsp_ga2opt_mpi_island && run --mpi=pmix -N 8 -n 8 -- ./tsp_ga2opt_mpi_island mu1979.tsp # Select mu1979.tsp  (Oman - 1,979 Cities)
```

### Run 2-opt 
Run the brute force approach (Correct/Exact Answer):
```bash
make tsp_2opt && ./tsp_bf
```

### Run Brute Force
Run the brute force approach (Correct/Exact Answer):
```bash
make tsp_lkh && ./tsp_bf
```

### Clean
Remove build artifacts:
```bash
make clean
```


### 其他 

Dataset from:
[Solving TSPs](https://www.math.uwaterloo.ca/tsp/world/countries.html)