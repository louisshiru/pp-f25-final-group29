
C_FLAG := -std=c++11 -O2 -fopenmp -Ilib

# Genetic Algorithm
GA_SRC := dataloader.cpp genetic.cpp
GA_OBJ := $(GA_SRC:.cpp=.o)
GA_TARGET := tsp_ga

# Brute Force
BF_SRC := brute_force.cpp dataloader.cpp
BF_OBJ := $(BF_SRC:.cpp=.o)
BF_TARGET := tsp_bf

all: $(GA_TARGET) $(BF_TARGET)

$(GA_TARGET): $(GA_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(BF_TARGET): $(BF_OBJ)
	g++ $(C_FLAG) -o $@ $^

%.o: %.cpp
	g++ $(C_FLAG) -c $< -o $@

clean:
	rm -f *.o $(GA_TARGET) $(BF_TARGET)
