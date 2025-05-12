#include "ac.h"
#include <cuda_runtime.h>
#include <cuda.h>

// Texture objects
cudaTextureObject_t tex_state_final;
cudaTextureObject_t tex_dfa;
cudaTextureObject_t tex_fail_state;

// Helper function to create texture object
cudaTextureObject_t createTextureObject(int* devPtr, size_t size) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = devPtr;
    resDesc.res.linear.desc = cudaCreateChannelDesc<int>();
    resDesc.res.linear.sizeInBytes = size;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    return texObj;
}

__global__ void profanity_filter_cuda(int* dfa, int* fail_state, unsigned char* sequences, 
                                     bool* valid_state, int offset, int num_sequences, int seq_length) {
    int num_sequences_per_block = num_sequences / gridDim.x;
    int num_sequences_per_thread = num_sequences / (gridDim.x * blockDim.x);

    int start = blockIdx.x * num_sequences_per_block + threadIdx.x * num_sequences_per_thread;
    int start_ptr = start * seq_length;

    int curr_state = 0;
    int idx = 0;
    int r_idx = 0;
    unsigned char base;

    while(r_idx < num_sequences_per_thread && (start + r_idx) < num_sequences) {
        base = sequences[start_ptr + (r_idx * seq_length) + idx++];

        if(base == '\n') {
            r_idx += 1;
            curr_state = 0;
            idx = 0;
            continue;
        }

        int ord = get_state_as_int(base);
        if(ord < 0) continue;

        while(curr_state != 0 && tex1Dfetch<int>(tex_dfa, curr_state * NUM_COLS + ord) == 0) {
            curr_state = tex1Dfetch<int>(tex_fail_state, curr_state);
        }

        if(curr_state != 0 || tex1Dfetch<int>(tex_dfa, curr_state * NUM_COLS + ord) != 0) {
            curr_state = tex1Dfetch<int>(tex_dfa, curr_state * NUM_COLS + ord);
            if(tex1Dfetch<int>(tex_state_final, curr_state)) {
                valid_state[start + r_idx] = true;
                break;
            }
        }
    }
}

void profanity_filter_parallel(int* dfa, int* fail_state, char* sequences, bool* valid_state, 
                              int num_sequences, int seq_length, int num_threads, int num_blocks) {
    if(num_sequences < num_blocks * num_threads) {
        num_blocks = 128;
        num_threads = num_sequences / num_blocks;
    }

    // Device pointers
    int* d_dfa = nullptr;
    int* d_fail_state = nullptr;
    unsigned char* d_sequences = nullptr;
    bool* d_valid_state = nullptr;
    int* s_final = nullptr;
    int* final = (int*)malloc(NUM_ROWS * sizeof(int));

    // Populate final states
    for(int i = 0; i < NUM_ROWS; i++) {
        final[i] = dfa[i * NUM_COLS + 0];
    }

    // Allocate device memory
    cudaMalloc((void**)&d_fail_state, NUM_ROWS * sizeof(int));
    cudaMalloc((void**)&d_valid_state, num_sequences * sizeof(bool));
    cudaMalloc((void**)&d_dfa, NUM_COLS * NUM_ROWS * sizeof(int));
    cudaMalloc((void**)&s_final, NUM_ROWS * sizeof(int));
    cudaMalloc((void**)&d_sequences, num_sequences * seq_length * sizeof(unsigned char));

    // Copy data to device
    cudaMemcpy(d_fail_state, fail_state, NUM_ROWS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dfa, dfa, NUM_ROWS * NUM_COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s_final, final, NUM_ROWS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sequences, sequences, num_sequences * seq_length * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_valid_state, 0, num_sequences * sizeof(bool));

    // Create texture objects
    tex_state_final = createTextureObject(s_final, NUM_ROWS * sizeof(int));
    tex_dfa = createTextureObject(d_dfa, NUM_ROWS * NUM_COLS * sizeof(int));
    tex_fail_state = createTextureObject(d_fail_state, NUM_ROWS * sizeof(int));

    // Launch kernel
    dim3 grid(num_blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);
    profanity_filter_cuda<<<grid, threads>>>(d_dfa, d_fail_state, d_sequences, d_valid_state, 
                                           0, num_sequences, seq_length);

    // Copy results back
    cudaMemcpy(valid_state, d_valid_state, num_sequences * sizeof(bool), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroyTextureObject(tex_state_final);
    cudaDestroyTextureObject(tex_dfa);
    cudaDestroyTextureObject(tex_fail_state);
    
    cudaFree(d_dfa);
    cudaFree(s_final);
    cudaFree(d_fail_state);
    cudaFree(d_sequences);
    cudaFree(d_valid_state);
    free(final);
}