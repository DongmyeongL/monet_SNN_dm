# Define the compiler
CC = nvcc

# Define compiler flags
CFLAGS = -O3 -arch=sm_70

# Define the target executable
TARGET = test

# Define the source files
SRCS = CUDA_DM_NEURON.cu CUDA_DM_RUN.cu main.cpp

# Define the object files
OBJS = $(SRCS:.c=.o)

# Default rule
all: $(TARGET)

# Rule for generating the target executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Rule for compiling source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)
