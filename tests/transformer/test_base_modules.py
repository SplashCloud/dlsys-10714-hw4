import sys
sys.path.append('./python')
sys.path.append('./transformer')

import numpy as np
import pytest
import torch
import math

import needle as ndl
from transformer.base_modules import (
    Linear, Embedding, RMSNorm, SwiGLU, RoPE, MultiHeadSelfAttention,
    softmax, scaled_dot_product_attention
)

_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


# Test Linear
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("in_features", [8, 16])
@pytest.mark.parametrize("out_features", [8, 16])
def test_linear_forward(device, batch_size, in_features, out_features):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, in_features).astype(np.float32)
    
    # Needle implementation
    model_ndl = Linear(in_features, out_features, device=device)
    x_ndl = ndl.Tensor(x_np, device=device)
    out_ndl = model_ndl(x_ndl)
    
    # PyTorch implementation
    model_torch = torch.nn.Linear(in_features, out_features, bias=False)
    model_torch.weight.data = torch.tensor(model_ndl.W.cached_data.numpy().T)
    x_torch = torch.tensor(x_np, requires_grad=True)
    out_torch = model_torch(x_torch)
    
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("in_features", [8, 16])
@pytest.mark.parametrize("out_features", [8, 16])
def test_linear_backward(device, batch_size, in_features, out_features):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, in_features).astype(np.float32)
    
    # Needle implementation
    model_ndl = Linear(in_features, out_features, device=device)
    x_ndl = ndl.Tensor(x_np, device=device, requires_grad=True)
    out_ndl = model_ndl(x_ndl)
    loss_ndl = out_ndl.sum()
    loss_ndl.backward()
    
    # PyTorch implementation
    model_torch = torch.nn.Linear(in_features, out_features, bias=False)
    model_torch.weight.data = torch.tensor(model_ndl.W.cached_data.numpy().T)
    x_torch = torch.tensor(x_np, requires_grad=True)
    out_torch = model_torch(x_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()
    
    np.testing.assert_allclose(x_ndl.grad.numpy(), x_torch.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(model_ndl.W.grad.numpy().T, model_torch.weight.grad.numpy(), atol=1e-5, rtol=1e-5)


# Test Embedding
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("num_embeddings", [10, 20])
@pytest.mark.parametrize("embedding_dim", [8, 16])
def test_embedding_forward(device, batch_size, seq_len, num_embeddings, embedding_dim):
    np.random.seed(0)
    token_ids_np = np.random.randint(0, num_embeddings, size=(batch_size, seq_len)).astype(np.int64)
    
    # Needle implementation
    model_ndl = Embedding(num_embeddings, embedding_dim, device=device)
    token_ids_ndl = ndl.Tensor(token_ids_np, device=device)
    out_ndl = model_ndl(token_ids_ndl)
    
    # PyTorch implementation
    model_torch = torch.nn.Embedding(num_embeddings, embedding_dim)
    model_torch.weight.data = torch.tensor(model_ndl.vocab.cached_data.numpy())
    token_ids_torch = torch.tensor(token_ids_np, dtype=torch.long)
    out_torch = model_torch(token_ids_torch)
    
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("num_embeddings", [10, 20])
@pytest.mark.parametrize("embedding_dim", [8, 16])
def test_embedding_backward(device, batch_size, seq_len, num_embeddings, embedding_dim):
    np.random.seed(0)
    token_ids_np = np.random.randint(0, num_embeddings, size=(batch_size, seq_len)).astype(np.int64)
    
    # Needle implementation
    model_ndl = Embedding(num_embeddings, embedding_dim, device=device)
    token_ids_ndl = ndl.Tensor(token_ids_np, device=device)
    out_ndl = model_ndl(token_ids_ndl)
    loss_ndl = out_ndl.sum()
    loss_ndl.backward()
    
    # PyTorch implementation
    model_torch = torch.nn.Embedding(num_embeddings, embedding_dim)
    model_torch.weight.data = torch.tensor(model_ndl.vocab.cached_data.numpy())
    token_ids_torch = torch.tensor(token_ids_np, dtype=torch.long)
    out_torch = model_torch(token_ids_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()

    np.testing.assert_allclose(model_ndl.vocab.grad.numpy(), model_torch.weight.grad.numpy(), atol=1e-5, rtol=1e-5)


# Test RMSNorm
def rms_norm_torch(x, g, eps=1e-5):
    """PyTorch implementation of RMSNorm"""
    x_float32 = x.float()
    d_model = x.shape[-1]
    # RMSNorm: sqrt(mean(x^2) + eps) = sqrt(sum(x^2) / d_model + eps)
    rms = torch.sqrt(torch.sum(x_float32 ** 2, dim=-1, keepdim=True) / d_model + eps)
    return (x_float32 / rms * g).to(x.dtype)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_model", [8, 16])
def test_rmsnorm_forward(device, batch_size, seq_len, d_model):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    
    # Needle implementation
    model_ndl = RMSNorm(d_model, device=device)
    x_ndl = ndl.Tensor(x_np, device=device)
    out_ndl = model_ndl(x_ndl)
    
    # PyTorch implementation
    x_torch = torch.tensor(x_np, requires_grad=True)
    g_torch = torch.tensor(model_ndl.g.cached_data.numpy(), requires_grad=True)
    out_torch = rms_norm_torch(x_torch, g_torch)
    
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_model", [8, 16])
def test_rmsnorm_backward(device, batch_size, seq_len, d_model):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    
    # Needle implementation
    model_ndl = RMSNorm(d_model, device=device)
    x_ndl = ndl.Tensor(x_np, device=device, requires_grad=True)
    out_ndl = model_ndl(x_ndl)
    loss_ndl = out_ndl.sum()
    
    print(f"\n{'='*80}")
    print("NEEDLE FORWARD PASS")
    print(f"{'='*80}")
    print(f"Loss: {loss_ndl.numpy()}")
    
    loss_ndl.backward()
    
    # PyTorch implementation
    x_torch = torch.tensor(x_np, requires_grad=True)
    g_torch = torch.tensor(model_ndl.g.cached_data.numpy(), requires_grad=True)
    out_torch = rms_norm_torch(x_torch, g_torch)
    loss_torch = out_torch.sum()
    
    print(f"\n{'='*80}")
    print("PYTORCH FORWARD PASS")
    print(f"{'='*80}")
    print(f"Loss: {loss_torch.detach().numpy()}")
    
    loss_torch.backward()

    np.testing.assert_allclose(model_ndl.g.grad.numpy(), g_torch.grad.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(x_ndl.grad.numpy(), x_torch.grad.numpy(), atol=1e-4, rtol=1e-4)


# Test SwiGLU
def swiglu_torch(x, w1, w2, w3):
    """PyTorch implementation of SwiGLU"""
    apply_w1 = x @ w1
    apply_swi = apply_w1 * torch.sigmoid(apply_w1)
    apply_w3 = x @ w3
    ele_wise_multiply = apply_swi * apply_w3
    apply_w2 = ele_wise_multiply @ w2
    return apply_w2


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("d_ff", [16, 32])
def test_swiglu_forward(device, batch_size, seq_len, d_model, d_ff):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    
    # Needle implementation
    model_ndl = SwiGLU(d_model, d_ff, device=device)
    x_ndl = ndl.Tensor(x_np, device=device)
    out_ndl = model_ndl(x_ndl)
    
    # PyTorch implementation
    x_torch = torch.tensor(x_np, requires_grad=True)
    w1_torch = torch.tensor(model_ndl.W1.cached_data.numpy(), requires_grad=True)
    w2_torch = torch.tensor(model_ndl.W2.cached_data.numpy(), requires_grad=True)
    w3_torch = torch.tensor(model_ndl.W3.cached_data.numpy(), requires_grad=True)
    out_torch = swiglu_torch(x_torch, w1_torch, w2_torch, w3_torch)
    
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("d_ff", [16, 32])
def test_swiglu_backward(device, batch_size, seq_len, d_model, d_ff):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    
    # Needle implementation
    model_ndl = SwiGLU(d_model, d_ff, device=device)
    x_ndl = ndl.Tensor(x_np, device=device, requires_grad=True)
    out_ndl = model_ndl(x_ndl)
    loss_ndl = out_ndl.sum()
    loss_ndl.backward()
    
    # PyTorch implementation
    x_torch = torch.tensor(x_np, requires_grad=True)
    w1_torch = torch.tensor(model_ndl.W1.cached_data.numpy(), requires_grad=True)
    w2_torch = torch.tensor(model_ndl.W2.cached_data.numpy(), requires_grad=True)
    w3_torch = torch.tensor(model_ndl.W3.cached_data.numpy(), requires_grad=True)
    out_torch = swiglu_torch(x_torch, w1_torch, w2_torch, w3_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()
    
    np.testing.assert_allclose(x_ndl.grad.numpy(), x_torch.grad.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(model_ndl.W1.grad.numpy(), w1_torch.grad.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(model_ndl.W2.grad.numpy(), w2_torch.grad.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(model_ndl.W3.grad.numpy(), w3_torch.grad.numpy(), atol=1e-4, rtol=1e-4)


# Test RoPE
def rope_torch(x, token_positions, theta, d_k, max_seq_len):
    """PyTorch implementation of RoPE"""
    batch_size, n_heads, seq_len, d_k = x.shape
    device = x.device
    
    # Generate rotary matrices
    rotary_matrices = []
    for i in range(max_seq_len):
        rotary_matrix_for_i = []
        for k in range(1, d_k // 2 + 1):
            theta_val = i * theta ** (-2 * (k - 1) / d_k)
            rot_mat = torch.tensor([
                [math.cos(theta_val), -math.sin(theta_val)],
                [math.sin(theta_val), math.cos(theta_val)]
            ], device=device, dtype=x.dtype)
            rotary_matrix_for_i.append(rot_mat)
        
        # Block diagonal matrix
        if len(rotary_matrix_for_i) == 1:
            block_diag_mat = rotary_matrix_for_i[0]
        else:
            block_diag_mat = torch.block_diag(*rotary_matrix_for_i)
        rotary_matrices.append(block_diag_mat)
    
    rotary_matrices = torch.stack(rotary_matrices, dim=0)  # (max_seq_len, d_k, d_k)
    
    # Apply RoPE: similar to needle implementation
    # rotary_matrix: (batch_size, seq_len, d_k, d_k) from indexing
    rotary_matrix = rotary_matrices[token_positions]  # (batch_size, seq_len, d_k, d_k)
    rotary_matrix = rotary_matrix.unsqueeze(1)  # (batch_size, 1, seq_len, d_k, d_k)
    
    # Matrix multiplication: (batch_size, 1, seq_len, d_k, d_k) @ (batch_size, n_heads, seq_len, d_k)
    # Need to expand x to (batch_size, n_heads, seq_len, d_k, 1) for matmul
    x_expanded = x.unsqueeze(-1)  # (batch_size, n_heads, seq_len, d_k, 1)
    # Broadcast rotary_matrix: (batch_size, 1, seq_len, d_k, d_k) -> (batch_size, n_heads, seq_len, d_k, d_k)
    rotary_matrix_expanded = rotary_matrix.expand(batch_size, n_heads, seq_len, d_k, d_k)
    output = (rotary_matrix_expanded @ x_expanded).squeeze(-1)  # (batch_size, n_heads, seq_len, d_k)
    
    return output


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_heads", [2, 4])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_k", [8, 16])
@pytest.mark.parametrize("theta", [10000.0])
@pytest.mark.parametrize("max_seq_len", [20])
def test_rope_forward(device, batch_size, n_heads, seq_len, d_k, theta, max_seq_len):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, n_heads, seq_len, d_k).astype(np.float32)
    token_positions_np = np.random.randint(0, max_seq_len, size=(batch_size, seq_len)).astype(np.int64)
    
    # Needle implementation
    model_ndl = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
    x_ndl = ndl.Tensor(x_np, device=device)
    token_positions_ndl = ndl.Tensor(token_positions_np, device=device)
    out_ndl = model_ndl(x_ndl, token_positions_ndl)
    
    # PyTorch implementation
    x_torch = torch.tensor(x_np, requires_grad=True)
    token_positions_torch = torch.tensor(token_positions_np, dtype=torch.long)
    out_torch = rope_torch(x_torch, token_positions_torch, theta, d_k, max_seq_len)
    
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), atol=1e-4, rtol=1e-4)


# Test softmax
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape", [(4, 8), (2, 5, 10)])
@pytest.mark.parametrize("dim", [0, -1])
def test_softmax(device, shape, dim):
    np.random.seed(0)
    x_np = np.random.randn(*shape).astype(np.float32)
    
    # Needle implementation
    x_ndl = ndl.Tensor(x_np, device=device)
    out_ndl = softmax(x_ndl, dim=dim)
    
    # PyTorch implementation
    x_torch = torch.tensor(x_np, requires_grad=True)
    out_torch = torch.softmax(x_torch, dim=dim)
    
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), atol=1e-5, rtol=1e-5)


# Test scaled_dot_product_attention
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_heads", [2, 4])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_k", [8, 16])
def test_scaled_dot_product_attention(device, batch_size, n_heads, seq_len, d_k):
    np.random.seed(0)
    Q_np = np.random.randn(batch_size, n_heads, seq_len, d_k).astype(np.float32)
    K_np = np.random.randn(batch_size, n_heads, seq_len, d_k).astype(np.float32)
    V_np = np.random.randn(batch_size, n_heads, seq_len, d_k).astype(np.float32)
    
    # Create causal mask
    mask_np = np.tril(np.ones((seq_len, seq_len))).astype(bool)
    mask_np = np.broadcast_to(mask_np, (batch_size, n_heads, seq_len, seq_len))
    
    # Needle implementation
    Q_ndl = ndl.Tensor(Q_np, device=device)
    K_ndl = ndl.Tensor(K_np, device=device)
    V_ndl = ndl.Tensor(V_np, device=device)
    mask_ndl = ndl.Tensor(mask_np, device=device)
    out_ndl = scaled_dot_product_attention(Q_ndl, K_ndl, V_ndl, mask=mask_ndl)
    
    # PyTorch implementation
    Q_torch = torch.tensor(Q_np, requires_grad=True)
    K_torch = torch.tensor(K_np, requires_grad=True)
    V_torch = torch.tensor(V_np, requires_grad=True)
    
    # Our implementation: mask=True means attend, mask=False means mask out
    # PyTorch: attn_mask=True means mask out, attn_mask=False means attend
    # So we need to invert the mask
    mask_torch = torch.tensor(mask_np, dtype=torch.bool)
    attn_mask = ~mask_torch  # Invert: True means mask out in PyTorch
    attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))
    attn_mask = attn_mask.masked_fill(~attn_mask, 0.0)
    
    out_torch = torch.nn.functional.scaled_dot_product_attention(
        Q_torch, K_torch, V_torch, 
        attn_mask=attn_mask,
        is_causal=False
    )
    
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), atol=1e-4, rtol=1e-4)


# Test MultiHeadSelfAttention
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_embedding", [16, 32])
@pytest.mark.parametrize("d_attn", [16, 32])
@pytest.mark.parametrize("num_heads", [2, 4])
def test_multihead_self_attention_forward(device, batch_size, seq_len, d_embedding, d_attn, num_heads):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, seq_len, d_embedding).astype(np.float32)
    
    # Needle implementation
    model_ndl = MultiHeadSelfAttention(
        d_embedding=d_embedding, 
        d_attn=d_attn, 
        num_heads=num_heads,
        device=device
    )
    x_ndl = ndl.Tensor(x_np, device=device)
    out_ndl = model_ndl(x_ndl)
    
    # Basic shape check
    assert out_ndl.shape == (batch_size, seq_len, d_attn)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_embedding", [16, 32])
@pytest.mark.parametrize("d_attn", [16, 32])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("theta", [10000.0])
@pytest.mark.parametrize("max_seq_len", [20])
def test_multihead_self_attention_with_rope(device, batch_size, seq_len, d_embedding, d_attn, num_heads, theta, max_seq_len):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, seq_len, d_embedding).astype(np.float32)
    token_positions_np = np.random.randint(0, max_seq_len, size=(batch_size, seq_len)).astype(np.int64)
    
    # Needle implementation
    model_ndl = MultiHeadSelfAttention(
        d_embedding=d_embedding, 
        d_attn=d_attn, 
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
        device=device
    )
    x_ndl = ndl.Tensor(x_np, device=device)
    token_positions_ndl = ndl.Tensor(token_positions_np, device=device)
    out_ndl = model_ndl(x_ndl, token_positions=token_positions_ndl)
    
    # Basic shape check
    assert out_ndl.shape == (batch_size, seq_len, d_attn)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [5, 10])
@pytest.mark.parametrize("d_embedding", [16, 32])
@pytest.mark.parametrize("d_attn", [16, 32])
@pytest.mark.parametrize("num_heads", [2, 4])
def test_multihead_self_attention_backward(device, batch_size, seq_len, d_embedding, d_attn, num_heads):
    np.random.seed(0)
    x_np = np.random.randn(batch_size, seq_len, d_embedding).astype(np.float32)
    
    # Needle implementation
    model_ndl = MultiHeadSelfAttention(
        d_embedding=d_embedding, 
        d_attn=d_attn, 
        num_heads=num_heads,
        device=device
    )
    x_ndl = ndl.Tensor(x_np, device=device, requires_grad=True)
    out_ndl = model_ndl(x_ndl)
    loss_ndl = out_ndl.sum()
    loss_ndl.backward()
    
    # Check gradients exist
    assert x_ndl.grad is not None
    assert model_ndl.WQ.grad is not None
    assert model_ndl.WK.grad is not None
    assert model_ndl.WV.grad is not None
    assert model_ndl.WO.grad is not None
