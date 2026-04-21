# CUDA Makefile for libturboquant_cuda.so
#
# Targets:
#   make          — build shared library
#   make test     — build and run GoogleTest tests
#   make clean    — remove build artifacts
#
# Configuration:
#   CUDA_HOME     — CUDA toolkit path (default: /usr/local/cuda)
#   CUDA_ARCH     — target architectures (default: sm_80;sm_90 for A100/H100)
#   BUILD_DIR     — output directory (default: lib)

CUDA_HOME  ?= /usr/local/cuda
NVCC       := $(CUDA_HOME)/bin/nvcc
CUDA_ARCH  ?= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90

BUILD_DIR  := lib
SRC_DIR    := src
INC_DIR    := include
TEST_DIR   := tests

# Compiler flags
NVCC_FLAGS := -std=c++17 -O3 --compiler-options '-fPIC -Wall' \
              $(CUDA_ARCH) \
              -I$(INC_DIR) \
              --expt-relaxed-constexpr

LDFLAGS    := -lcudart -lcublas -lcurand

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS))

# Library
LIB := $(BUILD_DIR)/libturboquant_cuda.so

.PHONY: all clean test

all: $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(wildcard $(INC_DIR)/*.h) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(LIB): $(OBJS)
	$(NVCC) -shared $(CUDA_ARCH) -o $@ $^ $(LDFLAGS)
	@echo "==> Built $(LIB)"

# GoogleTest
GTEST_DIR  ?= /usr/local
GTEST_INC  := $(GTEST_DIR)/include
GTEST_LIB  := $(GTEST_DIR)/lib

TEST_SRCS  := $(wildcard $(TEST_DIR)/*.cu) $(wildcard $(TEST_DIR)/*.cpp)
TEST_BIN   := $(BUILD_DIR)/test_turboquant

test: $(LIB) $(TEST_SRCS)
	@if [ -z "$(TEST_SRCS)" ]; then \
		echo "No test files found in $(TEST_DIR)/"; \
		exit 0; \
	fi
	$(NVCC) $(NVCC_FLAGS) \
		-I$(GTEST_INC) \
		$(TEST_SRCS) \
		-L$(BUILD_DIR) -lturboquant_cuda \
		-L$(GTEST_LIB) -lgtest -lgtest_main \
		$(LDFLAGS) -lpthread \
		-o $(TEST_BIN)
	LD_LIBRARY_PATH=$(BUILD_DIR):$(LD_LIBRARY_PATH) $(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
