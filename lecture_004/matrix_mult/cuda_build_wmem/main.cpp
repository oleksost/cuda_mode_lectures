#include <torch/extension.h>
torch::Tensor matmul_wmem(torch::Tensor A, torch::Tensor B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul_wmem", torch::wrap_pybind_function(matmul_wmem), "matmul_wmem");
}