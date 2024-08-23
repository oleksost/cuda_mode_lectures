from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
import torch


def compile_extension():
    cuda_source = Path("/home/mila/o/ostapeno/dev/lectures/lecture_004/matrix_mult/mat_mul_kernel.cu").read_text()
    cpp_source = "torch::Tensor matmul(torch::Tensor A, torch::Tensor B);"

    # Load the CUDA kernel as a PyTorch extension
    ext = load_inline(
        name="matmul",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        build_directory='/home/mila/o/ostapeno/dev/lectures/lecture_004/matrix_mult//cuda_build',
    )
    return ext

def compile_extension_wmem():
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

# n,m = 1000, 1000
ext = compile_extension()
ext_wmem = compile_extension_wmem()
# A = torch.randn(n,m).cuda()
# B = torch.randn(m,n).cuda()


def cuda_matmul(A,B):
    # ext = compile_extension()
    C = ext.matmul(A, B)
    return C
    # print("Output:", C.shape, C.dtype)
    # print("Output:", C)
    # assert torch.all(torch.isclose(torch.matmul(A, B), C))


def cuda_matmul_wmem(A,B):
    # ext = compile_extension()
    C = ext_wmem.matmul_wmem(A, B)
    return C
    # print("Output:", C.shape, C.dtype)
    # print("Output:", C)
    # assert torch.all(torch.isclose(torch.matmul(A, B), C))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'cuda-kernel',
            'cuda-kernel_wmem',
            'torch-native',
            'torch-compile'
        ],  # possible values for `line_arg`
        line_names=[
            "CUDA Kernel",
            "CUDA Kernel with memory optimization",
            "Torch (native)",
            "Torch (compiled)"
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('green', '--'), ('red', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="matrix multiplication performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):    
    A = torch.randn(M,N, device='cuda', dtype=torch.float32).cuda()
    B = torch.randn(N,M, device='cuda', dtype=torch.float32).cuda()
    nelement = M * N
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A,B), quantiles=quantiles)
    if provider == 'cuda-kernel':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cuda_matmul(A,B), quantiles=quantiles)
    if provider == 'cuda-kernel_wmem':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cuda_matmul_wmem(A,B), quantiles=quantiles)
    if provider == 'torch-compile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.compile(torch.matmul)(A,B), quantiles=quantiles)
    gbps = lambda ms: 3 * M * N * A.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path='/home/mila/o/ostapeno/dev/lectures/lecture_004/matrix_mult/matmul_report')
