from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
import torch
import time


def compile_extension():
    cuda_source = Path("/home/mila/o/ostapeno/dev/lectures/lecture_004/matrix_mult/mat_mul_kernel.cu").read_text()
    cpp_source = "torch::Tensor matmul_wmem(torch::Tensor A, torch::Tensor B);"

    # Load the CUDA kernel as a PyTorch extension
    ext = load_inline(
        name="matmul",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul_wmem"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        build_directory='/home/mila/o/ostapeno/dev/lectures/lecture_004/matrix_mult/cuda_build_wmem',
    )
    return ext

n,m = 56, 28
ext = compile_extension()
A = torch.randn(n,m).cuda()
B = torch.randn(m,n).cuda()


C = ext.matmul_wmem(A, B)



print("Output:", C.shape, C.dtype)
print("Output:", C)
assert torch.all(torch.isclose(torch.matmul(A, B), C, atol=1e-2))
