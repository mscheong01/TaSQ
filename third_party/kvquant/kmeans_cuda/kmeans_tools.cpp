#include <torch/extension.h>

template <int N>
torch::Tensor dist_argmin_half_batched(
    torch::Tensor A,
    torch::Tensor B
);

// Python 바인딩
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dist_argmin_half_batched_d4", &dist_argmin_half_batched<4>, "dist_argmin_half_batched_d4 (CUDA)");
    m.def("dist_argmin_half_batched_d8", &dist_argmin_half_batched<8>, "dist_argmin_half_batched_d8 (CUDA)");
    m.def("dist_argmin_half_batched_d9", &dist_argmin_half_batched<9>, "dist_argmin_half_batched_d9 (CUDA)");
    m.def("dist_argmin_half_batched_d10", &dist_argmin_half_batched<10>, "dist_argmin_half_batched_d10 (CUDA)");
}
