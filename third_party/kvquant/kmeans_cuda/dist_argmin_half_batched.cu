#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

constexpr int TILE_SIZE = 64;

// distArgminHalfBatchedKernel
template <int N>
__global__ void distArgminHalfBatchedKernel(
    const half* __restrict__ A,  // [L, N]
    const half* __restrict__ B,  // [M, N]
    int L,
    int M,
    int32_t* __restrict__ C
)
{
    int b = blockIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= L) return;

    half a_reg2[N];
    const half* A2 = reinterpret_cast<const half*>(A + b * L * N + row * N);
    #pragma unroll
    for(int i = 0; i < N; i++){
        a_reg2[i] = A2[i];
    }

    float minDist = 1e30f;
    int   minIdx  = -1;

    __shared__ half b_tile[N][TILE_SIZE];

    for(int tileStart = 0; tileStart < M; tileStart += TILE_SIZE){
        int tileRow = tileStart + threadIdx.x;

        if(tileRow < M) {
            const half* B2 = reinterpret_cast<const half*>(B + b * M * N + tileRow * N);
            #pragma unroll
            for(int k = 0; k < N; k++){
                b_tile[k][threadIdx.x] = B2[k];
            }
        }
        __syncthreads();

        #pragma unroll
        for(int t = 0; t < TILE_SIZE; t++){
            int b_idx = tileStart + t;
            if(b_idx >= M) break;

            float dist = 0.f;
            #pragma unroll
            for(int k = 0; k < N; k++){
                float a = __half2float(a_reg2[k]);
                float b = __half2float(b_tile[k][t]);
                float diff = a - b;
                dist += diff * diff;
            }
            if(dist < minDist){
                minDist = dist;
                minIdx  = b_idx;
            }
        }
        __syncthreads();
    }

    C[b * L + row] = minIdx;
}


template <int N>
torch::Tensor dist_argmin_half_batched(
    torch::Tensor A, // [B, L, N]
    torch::Tensor B // [B, M, N]
){
    TORCH_CHECK(A.is_contiguous());
    TORCH_CHECK(B.is_contiguous());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(A.dim() == 3 && A.size(2) == N);
    TORCH_CHECK(B.dim() == 3 && B.size(2) == N);
    int dev_index = A.device().index();
    cudaSetDevice(dev_index);


    int64_t BS = A.size(0);
    int64_t L = A.size(1);
    int64_t M = B.size(1);

    auto out_opts = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    auto C = torch::empty({BS, L}, out_opts);

    dim3 gridDim(BS, (L + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blockDim(TILE_SIZE);

    distArgminHalfBatchedKernel<N><<<gridDim, blockDim>>>(
        reinterpret_cast<const half*>(A.data_ptr()),
        reinterpret_cast<const half*>(B.data_ptr()),
        (int)L,
        (int)M,
        C.data_ptr<int32_t>()
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess);

    return C;
}

template torch::Tensor dist_argmin_half_batched<4>(
    torch::Tensor A,
    torch::Tensor B
);

template torch::Tensor dist_argmin_half_batched<8>(
    torch::Tensor A,
    torch::Tensor B
);

template torch::Tensor dist_argmin_half_batched<9>(
    torch::Tensor A,
    torch::Tensor B
);

template torch::Tensor dist_argmin_half_batched<10>(
    torch::Tensor A,
    torch::Tensor B
);
