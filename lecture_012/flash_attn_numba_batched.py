import os
import numba
import math
from numba import cuda
import numba.cuda
import torch
import pdb
from torch.autograd import Function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"


s = 1024 # seq length
n_heads = 8
d_model = 128 # hidden dim
d_head = d_model // n_heads # this is in numbers, each number is float32, so 4 bytes
assert d_model % n_heads == 0

b = 32 # batch size
sram = cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK # this is in bytes, e.g. 0xc000 = 49152 bytes
B_r = B_c = math.ceil(sram / (d_head * 4 * 4)) # because we need to store Q,K,V and O in SRAM <- the number of threads we can run in parallel given the SRAM size, this is also the number of threads per block

@cuda.jit
def flash_attn_kernel(q, k, v, out, T_r, T_c):
    # we run one block per head
    block_b = cuda.blockIdx.x # coordinate of the batch index
    block_h = cuda.blockIdx.y # coordinate of the head index
    # then we run B_r threads per block taking care of the s and d dimention
    thread_i = cuda.threadIdx.x
    
    
    # k,v,q are shape (b,h,s,d)
    K_shared = cuda.shared.array(shape=(B_c, d_head), dtype=numba.float32)
    V_shared = cuda.shared.array(shape=(B_c, d_head), dtype=numba.float32)
    Q_shared = cuda.shared.array(shape=(B_r, d_head), dtype=numba.float32)
    
    

    for j in range(T_r):
        r_idx = j * B_r + thread_i
        if r_idx < s:
            for c in range(Q_shared.shape[1]):
                Q_shared[thread_i, c] = q[block_b, block_h, r_idx, c]
        cuda.syncthreads()
        l, m = 0.0, 0.0 
        for i in range(T_c):
            c_idx = i * B_c + thread_i
            if c_idx < s:
                for c in range(K_shared.shape[1]):
                    K_shared[thread_i, c] = k[block_b, block_h, c_idx, c]
                for c in range(V_shared.shape[1]):
                    V_shared[thread_i, c] = v[block_b, block_h, c_idx, c]
            cuda.syncthreads()
            if r_idx < s:
                # Compute attention scores Sij = Qi * Kj^T
                for b_c in range(i * B_c, min((i + 1) * B_c, s)):
                    b_c = b_c - i * B_c
                    curr_l = l
                    curr_m = m
                    Sij = 0.0
                    for k_dim in range(Q_shared.shape[1]):
                        Sij += Q_shared[thread_i, k_dim] * K_shared[b_c, k_dim]

                    new_m = max(curr_m, Sij)
                    exp_Sij = math.exp(Sij - new_m)
                    exp_max = math.exp(curr_m - new_m)
                    new_l = curr_l * exp_max + exp_Sij

                    # Update output Oi += softmax * Vj
                    for v_dim in range(d_head):
                        # this writes each element to the global memory
                        out[block_b, block_h, r_idx, v_dim] = (
                            out[block_b, block_h, r_idx, v_dim] * (curr_l * exp_max / new_l)
                            + (exp_Sij / new_l) * V_shared[b_c, v_dim]
                        )

                    l = new_l
                    m = new_m
            cuda.syncthreads()

class FlashAttn(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        b,h,s,d_head_local = q.size()
        assert d_head_local == d_head
        # q is b, h, s, d
        # we parallelize by having a block of threads per head, and we have b x h heads
        grid_dim = (b, h)
        assert B_r == B_c
        block_dim = B_c
        # out should be b, h, s, d, where d is 
        out = torch.zeros(b, h, s, d_head).to(device)
        q_numba = numba.cuda.as_cuda_array(q.detach().float())
        k_numba = numba.cuda.as_cuda_array(k.detach().float())
        v_numba = numba.cuda.as_cuda_array(v.detach().float())
        out = numba.cuda.as_cuda_array(out)
        
        T_c,T_r = math.ceil(s/B_c), math.ceil(s/B_r)        
        flash_attn_kernel[grid_dim, block_dim](q_numba, k_numba, v_numba, out, T_r, T_c)
        cuda.synchronize()
        out_numba = out.copy_to_host()
        return torch.tensor(out_numba).to(device)
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented for FlashAttn")

class Attn(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, attn, attn_logits = attention(q, k, v)
        # Save intermediate results needed for backward pass
        ctx.save_for_backward(
            q,
            k,
            v,
            attn,
        )
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        print("backward")
        # Retrieve saved tensors
        q, k, v, attn = ctx.saved_tensors

        # Compute the gradient w.r.t. v
        grad_v = torch.matmul(attn.transpose(-2, -1), grad_output)

        # Compute the gradient w.r.t. attn (intermediate softmax gradient)
        grad_attn = torch.matmul(grad_output, v.transpose(-2, -1))

        # Gradient of softmax
        d_attn_logits = attn * (
            grad_attn - torch.sum(grad_attn * attn, dim=-1, keepdim=True)
        )

        # Compute the gradient w.r.t. q and k
        grad_q = torch.matmul(d_attn_logits, k)
        grad_k = torch.matmul(d_attn_logits.transpose(-2, -1), q)
        # pdb.set_trace()
        return grad_q, grad_k, grad_v


def attention(q, k, v, mask=None, dropout=None):
    # q,k,v : (b, h, s, d)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn = torch.nn.functional.softmax(attn_logits, dim=-1)  # (b, h, s, s)
    return torch.matmul(attn, v), attn, attn_logits


class Attention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head == d_head

        self.qkc_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, use_flash=False):
        b, s, _ = x.size()
        qkv = self.qkc_proj(x).reshape(b, s, self.n_heads, 3 * self.d_head)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = torch.split(qkv, self.d_head, dim=-1)  # (b, h, s, d)
        if use_flash:
            out_attn = FlashAttn.apply(q, k, v)
        else:
            out_attn = Attn.apply(q, k, v)
        out = out_attn.permute(0, 2, 1, 3).reshape(b, s, self.d_model)
        o = self.o_proj(out)
        return o, out_attn


def main():
    x = torch.randn(b, s, d_model).to(device)
    attention = Attention(d_model, n_heads).to(device)
    out, out_attn_fa = attention(x, use_flash=True)
    _, out_attn = attention(x, use_flash=False)
    print(torch.allclose(out_attn, out_attn_fa, atol=1e-2))  

if __name__ == "__main__":
    main()
