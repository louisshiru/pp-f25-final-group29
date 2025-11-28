
C_FLAG := -std=c++11 -O2 -fopenmp -Ilib

# Genetic Algorithm
GA_SRC := dataloader.cpp genetic.cpp
GA_OBJ := $(GA_SRC:.cpp=.o)
GA_TARGET := tsp_ga

# Brute Force
2OPT_SRC := 2-opt.cpp dataloader.cpp
2OPT_OBJ := $(2OPT_SRC:.cpp=.o)
2OPT_TARGET := tsp_2opt

# Brute Force
LKH_SRC := lkh.cpp dataloader.cpp
LKH_OBJ := $(LKH_SRC:.cpp=.o)
LKH_TARGET := tsp_lkh

# ACO
ACO_SRC := aco.cpp dataloader.cpp
ACO_OBJ := $(ACO_SRC:.cpp=.o)
ACO_TARGET := tsp_aco

all: $(GA_TARGET) $(2OPT_TARGET) $(LKH_TARGET) $(ACO_TARGET)

$(GA_TARGET): $(GA_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(2OPT_TARGET): $(2OPT_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(LKH_TARGET): $(LKH_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(ACO_TARGET): $(ACO_OBJ)
	g++ $(C_FLAG) -o $@ $^

%.o: %.cpp
	g++ $(C_FLAG) -c $< -o $@

clean:
	rm -f *.o $(GA_TARGET) $(2OPT_TARGET) $(LKH_TARGET) $(ACO_TARGET)
