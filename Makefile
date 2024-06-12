# Makefile

# Compiler
CXX = g++

# Compiler flags
CUDA_PATH = /usr/local/cuda
CUDNN_PATH = /usr/local/cudnn
CXXFLAGS = -std=c++11 -Wall -I$(CUDA_PATH)/include -I$(CUDNN_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH)/lib64 -lcudart -lcublas -lcudnn

# Executable names
EXEC = main
TEST_EXEC = test_attention

# Source files
SRCS = main.cpp attention.cpp attention_gpu.cpp
TEST_SRCS = test_attention.cpp attention.cpp attention_gpu.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# Header files
HEADERS = attention.h

# Default target
all: $(EXEC)

# Main executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Test executable
$(TEST_EXEC): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Pattern rules for object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $<

# Clean target to remove generated files
clean:
	rm -f $(EXEC) $(TEST_EXEC) $(OBJS) $(TEST_OBJS)

# Test target to compile and run tests
test: $(TEST_EXEC)
	./$(TEST_EXEC)

.PHONY: all clean test
