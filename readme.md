# PP-f25 Final Project: TSP with Genetic Algorithm

## Description
This project aims to solve the Traveling Salesman Problem (TSP) using a Genetic Algorithm. The primary objective is to apply parallel programming techniques to accelerate the computation.

We provide two solvers:
1.  **Genetic Algorithm (`tsp_ga`)**: A heuristic approach to find an approximate solution efficiently.
2.  **Brute Force (`tsp_bf`)**: An exact method to find the optimal solution, used for verification.

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

可以搜尋關鍵字 `[TODO]` 應該是優化這兩個地方就可以了 ?????