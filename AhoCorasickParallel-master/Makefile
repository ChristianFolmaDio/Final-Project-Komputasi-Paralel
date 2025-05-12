# Compiler configuration
NVCC := nvcc
CC := g++

# CUDA paths - sesuaikan dengan instalasi CUDA Anda
CUDA_PATH := C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9
CUDA_INCLUDEPATH := $(CUDA_PATH)/include

# Architecture flags (sesuaikan dengan GPU Anda)
CUDA_ARCH := -arch=sm_75  # Ubah sesuai GPU

# Compiler flags
NVCC_OPTS := -O3 $(CUDA_ARCH) -Xcompiler -Wall -std=c++14
GCC_OPTS := -O3 -m64 -Wall -std=c++14 -I"$(CUDA_INCLUDEPATH)"

# Target executable
TARGET := acp.exe

# Source files
CUDA_SOURCES := ac.cu
CPP_SOURCES := main.cpp ac_utils.cpp ac_serial.cpp ac_open_acc.cpp

# Object files
OBJECTS := $(CPP_SOURCES:.cpp=.obj) $(CUDA_SOURCES:.cu=.obj)

# Clean command for Windows
clean:
		del /Q $(TARGET) $(OBJECTS) 2> nul

# Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
		$(NVCC) $(NVCC_OPTS) -o $(TARGET) $(OBJECTS) -L"$(CUDA_PATH)/lib/x64" -lcudart

%.obj: %.cpp
		$(CC) $(GCC_OPTS) -c $< -o $@

%.obj: %.cu
		$(NVCC) $(NVCC_OPTS) -c $< -o $@
