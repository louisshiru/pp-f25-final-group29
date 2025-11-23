
C_FLAG := -std=c++11 -O2 -fopenmp -Ilib
SRC := main.cpp dataloader.cpp genetic.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := tsp_ga

$(TARGET): $(OBJ)
	g++ $(C_FLAG) -o $@ $^

%.o: %.cpp
	g++ $(C_FLAG) -c $< -o $@

all: $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
