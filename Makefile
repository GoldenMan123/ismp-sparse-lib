NVCC = nvcc
CUDA_PATH = $(shell which $(NVCC) | sed s@/bin/nvcc@@)

CUDA_ARCH = 35

CUDA_INCLUDE := -I$(CUDA_PATH)/include/

CXXWARN := -Wall -Wno-sign-compare
CXXOPT  := -O2 -fopenmp -std=c++0x
CXXFLAGS := $(CXXOPT) -g $(CXXWARN) -fPIC -I. $(CUDA_INCLUDE)

NVCCFLAGS := -O2 -v -I. -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	-DCUDA_ARCH=$(CUDA_ARCH) \
	--ptxas-options -v \
	--cudafe-options --diag_suppress=code_is_unreachable \
	--compiler-options -fPIC

LIB = libispm0-pic.a
TEST = run_test
all: $(LIB) $(TEST)

SPMV_OBJS  = cuda/spmv/dispatch-float-float.o cuda/spmv/dispatch-double-float.o cuda/spmv/dispatch-double-double.o
EXTRA_OBJS = util/cuda/sblas.o fastainv/fastainv.o util/cuda/initialize.o fsai/fsai.o cgSolver/cgSolver.o
TEST_OBJS = test/test.o fsai/fsai.o cgSolver/cgSolver.o
OBJS = $(SPMV_OBJS) $(EXTRA_OBJS)

$(LIB): $(OBJS)
	ar cr $@ $^

$(TEST): $(TEST_OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) -lcudart

cuda/spmv/dispatch-%.o: cuda/spmv/dispatch-%.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) --compiler-options -fpermissive
	objcopy --localize-hidden $@

util/cuda/%.o: util/cuda/%.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

fastainv/fastainv.o: fastainv/fastainv.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)
    
fsai/fsai.o: fsai/fsai.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

cgSolver/cgSolver.o: cgSolver/cgSolver.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

test/test.o: test/test.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

clean:
	-rm -f $(OBJS) $(LIB)
