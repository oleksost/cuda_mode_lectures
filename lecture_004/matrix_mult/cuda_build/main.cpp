#include <torch/extension.h>
torch::Tensor matmul(torch::Tensor A, torch::Tensor B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul", torch::wrap_pybind_function(matmul), "matmul");
}