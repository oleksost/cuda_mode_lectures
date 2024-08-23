#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define TILE_WIDTH 16

__global__
void matmul_kernel(float* output, float* A, float* B, int width, int height, int inne_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float value = 0.0f;
    if (col < width && row < height) {
        for (int i = 0; i < inne_dim; i++) {
            value += A[row * inne_dim + i] * B[i * width + col];
        } 
        output[row * width + col] = value;
    }
}


__global__
void matmul_kernel_wmem(float* output, float* A, float* B, int width, int height, int inne_dim) {   

    __shared__ float mA[TILE_WIDTH][TILE_WIDTH]; // block shared memory
    __shared__ float mB[TILE_WIDTH][TILE_WIDTH];

    int r = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int c = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float value = 0.0f;
    int num_phases = static_cast<int>(ceil(static_cast<float>(inne_dim) / TILE_WIDTH));
    
    for (int ph = 0; ph < num_phases; ++ph) {
        // Load data into shared memory with boundary checking
        if (r < height && (ph * TILE_WIDTH + tx) < inne_dim) {
            mA[ty][tx] = A[r * inne_dim + ph * TILE_WIDTH + tx];
        } else {
            mA[ty][tx] = 0.0f;
        }

        if (c < width && (ph * TILE_WIDTH + ty) < inne_dim) {
            mB[ty][tx] = B[(ph * TILE_WIDTH + ty) * width + c];
        } else {
            mB[ty][tx] = 0.0f;
        }

        __syncthreads(); // wait for all threads to finish copying, this syncs only within the same block
        
        // Perform the multiplication and accumulation
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += mA[ty][i] * mB[i][tx];
        }
        __syncthreads(); // ensure all computations are done before next phase
    }

    // Write the result to the output matrix
    if (r < height && c < width) {
        output[r * width + c] = value;
    }    
}



// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor matmul_wmem(torch::Tensor A, torch::Tensor B) {
    assert(A.device().type() == torch::kCUDA);
    assert(B.device().type() == torch::kCUDA);
    assert(A.size(1) == B.size(0));

    const auto height = A.size(0);
    const auto width = B.size(1);
    const auto inne_dim = A.size(1);

    auto result = torch::empty({height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    
    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),
        cdiv(height, threads_per_block.y)
    );

    matmul_kernel_wmem<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        width,
        height,
        inne_dim
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}


torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    assert(A.device().type() == torch::kCUDA);
    assert(B.device().type() == torch::kCUDA);
    assert(A.size(1) == B.size(0));

    const auto height = A.size(0);
    const auto width = B.size(1);
    const auto inne_dim = A.size(1);

    auto result = torch::empty({height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),
        cdiv(height, threads_per_block.y)
    );

    matmul_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        width,
        height,
        inne_dim
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
