
C_FLAG := -std=c++11 -O2 -fopenmp -Ilib -pg

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

# GA + 2-opt Hybrid
GA2_SRC := ga_2opt.cpp dataloader.cpp
GA2_OBJ := $(GA2_SRC:.cpp=.o)
GA2_TARGET := tsp_ga2opt

GA2_OMP_SRC := ga_2opt_omp.cpp dataloader.cpp
GA2_OMP_OBJ := $(GA2_OMP_SRC:.cpp=.o)
GA2_OMP_TARGET := tsp_ga2opt_omp

# GA + 2-opt cuda
# Host-side GA implementation in C++ and CUDA kernels in separate .cu
GA2_CUDA_SRC := dataloader.cpp ga_2opt_cuda.cpp
GA2_CUDA_CU  := kernel.cu
GA2_CUDA_OBJ := $(GA2_CUDA_SRC:.cpp=.o) $(GA2_CUDA_CU:.cu=.o)
GA2_CUDA_TARGET := tsp_ga2opt_cuda
GA2_CUDA_FLAG := -Ilib -std=c++11 -O2 -arch=sm_61 -lgomp

# Enable CUDA_HYBRID_OMP only when requested, e.g.:
#   make GA2_CUDA_TARGET CUDA_HYBRID_OMP=True
ifeq ($(CUDA_HYBRID_OMP),True)
C_FLAG += -DCUDA_HYBRID_OMP
endif

GPROF_RESULT := gmon.out profiling_result

all: $(GA_TARGET) $(2OPT_TARGET) $(LKH_TARGET) $(ACO_TARGET) $(GA2_TARGET) $(GA2_OMP_TARGET) $(GA2_CUDA_TARGET)

$(GA_TARGET): $(GA_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(2OPT_TARGET): $(2OPT_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(LKH_TARGET): $(LKH_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(ACO_TARGET): $(ACO_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(GA2_TARGET): $(GA2_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(GA2_OMP_TARGET): $(GA2_OMP_OBJ)
	g++ $(C_FLAG) -o $@ $^

$(GA2_CUDA_TARGET): $(GA2_CUDA_OBJ) 
	nvcc $(GA2_CUDA_FLAG) -o $@ $^

%.o: %.cu
	nvcc $(GA2_CUDA_FLAG) -c $< -o $@

%.o: %.cpp
	g++ $(C_FLAG) -c $< -o $@

clean:
	rm -f *.o $(GA_TARGET) $(2OPT_TARGET) $(LKH_TARGET) $(ACO_TARGET) $(GA2_TARGET) $(GA2_OMP_TARGET) $(GA2_CUDA_TARGET) $(GPROF_RESULT)
