
import sys
import os
# Add python directory to path if not already there
# Get the directory containing this file and add python directory relative to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
python_path = os.path.join(project_root, 'python')
if python_path not in sys.path and os.path.exists(python_path):
    sys.path.insert(0, python_path)

import math
from math import inf
import needle as ndl

sigmoid = ndl.nn.Sigmoid()

class Linear(ndl.nn.Module):

    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = ndl.nn.Parameter(ndl.init.xavier_normal(fan_in=self.in_features, fan_out=self.out_features, device=self.device, dtype=self.dtype))

    def forward(self, input: ndl.Tensor) -> ndl.Tensor:
        assert input.shape[-1] == self.in_features
        return input @ self.W


class Embedding(ndl.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        weight = ndl.init.randn(self.num_embeddings, self.embedding_dim, mean=0.0, std=1.0, device=self.device, dtype=self.dtype)
        self.vocab = ndl.nn.Parameter(array=weight, requires_grad=True)

    def forward(self, token_ids: ndl.Tensor) -> ndl.Tensor:
        return self.vocab[token_ids] # will change dtype into int64 inside

class RMSNorm(ndl.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = ndl.nn.Parameter(array=ndl.init.ones(self.d_model, dtype=dtype, device=device), requires_grad=True)

    def forward(self, x: ndl.Tensor) -> ndl.Tensor:
        assert x.shape[-1] == self.d_model
        # RMSNorm: sqrt(mean(x^2) + eps) = sqrt(sum(x^2) / d_model + eps)
        x_squared_sum = (x ** 2).sum(axes=-1, keepdims=True)
        rms = ((x_squared_sum / self.d_model + self.eps) ** 0.5)
        x_norm = x / rms
        return x_norm * self.g


class SwiGLU(ndl.nn.Module):

    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.W1 = ndl.nn.Parameter(ndl.init.xavier_normal(fan_in=self.d_model, fan_out=self.d_ff, device=self.device, dtype=self.dtype))
        self.W2 = ndl.nn.Parameter(ndl.init.xavier_normal(fan_in=self.d_ff, fan_out=self.d_model, device=self.device, dtype=self.dtype))
        self.W3 = ndl.nn.Parameter(ndl.init.xavier_normal(fan_in=self.d_model, fan_out=self.d_ff, device=self.device, dtype=self.dtype))

    def forward(self, input: ndl.Tensor) -> ndl.Tensor:
        assert input.shape[-1] == self.d_model
        apply_W1 = input @ self.W1 # ... d_model, d_model d_ff -> ... d_ff
        apply_swi = apply_W1 * sigmoid(apply_W1) # ... d_ff, ... d_ff -> ... d_ff
        apply_W3 = input @ self.W3 # ... d_model, d_model d_ff -> ... d_ff
        ele_wise_multiply = apply_swi * apply_W3 # ... d_ff, ... d_ff -> ... d_ff
        apply_W2 = ele_wise_multiply @ self.W2 # ... d_ff, d_ff d_model -> ... d_model
        return apply_W2


class RoPE(ndl.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self._register_rotary_matrices() # generate and save the rotary matrices (max_seq_len, d_k, d_k)

    def forward(self, x: ndl.Tensor, token_positions: ndl.Tensor) -> ndl.Tensor:
        # x.shape = (batch_size, n_heads, seq_len, d_k)
        # token_positions = (batch_size, seq_len)
        assert x.shape[-1] == self.d_k
        assert x.shape[-2] == token_positions.shape[-1] and x.shape[-2] <= self.max_seq_len
        # assert token_positions.dtype == "int64" or token_positions.dtype == "long"
        rotary_matrix: ndl.Tensor = self.rotary_matrices[token_positions] # (batch_size, seq_len, d_k1, d_k2)
        rotary_matrix = rotary_matrix.unsqueeze(1) # (batch_size, 1, seq_len, d_k1, d_k2)
        x = x.unsqueeze(-1) # (batch_size, n_heads, seq_len, d_k, 1)
        result = rotary_matrix @ x # (batch_size, 1, seq_len, d_k, d_k) @ (batch_size, n_heads, seq_len, d_k, 1)
        return result.squeeze(-1)

    def _register_rotary_matrices(self):
        rotary_matrices = []
        for i in range(self.max_seq_len):
            rotary_matrix_for_i = []
            for k in range(1, self.d_k//2 + 1):
                theta = i * self.theta ** (-2*(k-1)/self.d_k)
                rotary_matrix_for_i.append(ndl.Tensor([[math.cos(theta), -math.sin(theta)],
                                            [math.sin(theta), math.cos(theta)]], device=self.device))
            rotary_matrices.append(ndl.block_diag(*rotary_matrix_for_i))
        self.register_buffer("rotary_matrices", ndl.stack(rotary_matrices, axis=0).to(device=self.device))


class MultiHeadSelfAttention(ndl.nn.Module):

    def __init__(self, d_embedding: int, d_attn: int,num_heads: int,
                       theta: float = 0, max_seq_len: int = 0, device = None, dtype = None):
        super().__init__()
        self.d_embedding = d_embedding
        self.d_attn = d_attn
        self.num_heads = num_heads
        self.d_k = d_attn // num_heads
        self.d_v = d_attn // num_heads
        self.device = device
        self.dtype = dtype

        self.WQ = ndl.nn.Parameter(ndl.init.xavier_normal(fan_in=self.d_embedding, fan_out=self.num_heads*self.d_k, device=self.device, dtype=self.dtype))
        self.WK = ndl.nn.Parameter(ndl.init.xavier_normal(fan_in=self.d_embedding, fan_out=self.num_heads*self.d_k, device=self.device, dtype=self.dtype))
        self.WV = ndl.nn.Parameter(ndl.init.xavier_normal(fan_in=self.d_embedding, fan_out=self.num_heads*self.d_v, device=self.device, dtype=self.dtype))
        self.WO = ndl.nn.Parameter(ndl.init.xavier_normal(fan_in=self.num_heads*self.d_v, fan_out=self.d_attn, device=self.device, dtype=self.dtype))

        self.enable_rope = ((theta != 0) and (max_seq_len != 0))
        if self.enable_rope:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=self.device)

    def forward(self, x: ndl.Tensor, token_positions: ndl.Tensor | None = None) -> ndl.Tensor:
        assert x.shape[-1] == self.d_embedding # x.shape = (batch_size, ..., seq_len, d_embedding)
        batch_size, seq_len, d_model = x.shape
        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV
        Q = Q.reshape(shape=(batch_size, self.num_heads, seq_len, self.d_k))
        K = K.reshape(shape=(batch_size, self.num_heads, seq_len, self.d_k))
        V = V.reshape(shape=(batch_size, self.num_heads, seq_len, self.d_v))
        if self.enable_rope:
            assert token_positions is not None
            Q = self.rope.forward(x=Q, token_positions=token_positions)
            K = self.rope.forward(x=K, token_positions=token_positions)
        mask = ndl.tril(ndl.ones(x.shape[-2], x.shape[-2], device=x.device))
        Attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        Attn = Attn.reshape(shape=(batch_size, seq_len, self.num_heads * self.d_v))
        return Attn @ self.WO

## Functions

def softmax(x: ndl.Tensor, dim: int) -> ndl.Tensor:
    x_transpose = x.transpose((dim, len(x.shape)-1))
    x_max = ndl.maximum(x_transpose, axes=-1, keepdims=True)
    x_norm_exp = ndl.exp(x_transpose - x_max)
    x_norm_exp_sum = ndl.summation(x_norm_exp, axes=-1, keepdims=True)
    x_softmax = x_norm_exp / x_norm_exp_sum
    return x_softmax.transpose((dim, len(x.shape)-1))

def scaled_dot_product_attention(Q: ndl.Tensor, K: ndl.Tensor, V: ndl.Tensor, mask: ndl.Tensor | None = None) -> ndl.Tensor:
    QK = Q @ K.transpose()
    scaled_QK = QK / math.sqrt(Q.shape[-1])
    if mask is not None:
        masked_matrix = ndl.init.constant(*mask.shape, c=-inf, dtype=Q.dtype, device=Q.device)
        masked_matrix[mask] = 0
        scaled_QK += masked_matrix
    softmax_scaled_QK = softmax(scaled_QK, dim=-1)
    return softmax_scaled_QK @ V