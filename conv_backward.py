import torch
import numpy as np


N = 1
H, W, K = 9, 9, 3
IC, OC = 1, 1
P, S = 0, 1
NH, NW = (H+2*P-K)//S+1, (W+2*P-K)//S+1

# NH+2*P1 - K + 1 = H
# 2*P-K+1+2*P1-K+1=0
# P1=2*K-2*P-2

random = True

def dilate(a: np.array, axes: tuple, dilation: int, full=False):
    new_shape = list(a.shape)
    origin_region_indices = [slice(0, dim, 1) for dim in a.shape]
    for axis in axes:
        if full:
            new_shape[axis] *= (dilation + 1)
        else:
            new_shape[axis] += dilation * (new_shape[axis]-1)
        origin_region_indices[axis] = slice(0, new_shape[axis], dilation + 1)
    result = np.zeros(tuple(new_shape))
    result[tuple(origin_region_indices)] = a
    return result

def torch_conv_backward(Z_, W_):
    '''
    Z: torch.Tensor (N, H, W, IC)
    W: torch.Tensor (K, K, IC, OC)
    '''
    O = torch.nn.functional.conv2d(Z_.permute(0, 3, 1, 2), W_.permute(3, 2, 0, 1), padding=P, stride=S)
    print(f'O.shape: {O.shape}')
    O.sum().backward()
    return Z_.grad.numpy(), W_.grad.numpy()


def manual_conv_backward(Z_, W_, out_grad):
    '''
    Z: torch.Tensor (N, H, W, IC)
    W: torch.Tensor (K, K, IC, OC)
    out_grad: torch.Tensor (N, NH, NW, OC)

    1. 卷积核不能滑动到边界 (H+2P-K)//S 不能被整除
    2. dilate之后 
    '''
    # Z.grad
    if S > 1:
        out_grad = torch.Tensor(dilate(out_grad.detach().numpy(), (1,2), S-1, full=False))
    print(f'out_grad.shape: {out_grad.shape}')
    # NH + 2P - K + 1 = H
    # h_padding, w_padding = (H+K-1-out_grad.shape[1])//2, (H+K-1-out_grad.shape[2])//2
    h_padding, w_padding = K-P-1, K-P-1
    hr, wr = (H+2*P-K)%S, (W+2*P-K)%S
    print(f'hr: {H+2*P-K}%{S}={hr}, wr: {W+2*P-K}%{S}={wr}')
    out_grad = torch.Tensor(np.pad(out_grad.detach().numpy(), ((0,0),(h_padding,h_padding+hr),(w_padding,w_padding+wr),(0,0))))
    flip_W = np.flip(W_.permute(0,1,3,2).detach().numpy(), (0,1)).copy() # (K, K, OC, IC)
    Z_grad = torch.nn.functional.conv2d(out_grad.permute(0, 3, 1, 2), torch.Tensor(flip_W).permute(3, 2, 0, 1))
    Z_grad = Z_grad.permute(0, 2, 3, 1)
    # W_grad
    Z0 = Z_.permute(3, 1, 2, 0) # (IC, H, W, N) => (N', H, W, IC')
    Z0 = torch.Tensor(np.pad(Z0.detach().numpy(), ((0,0),(K-1,K-1),(K-1,K-1),(0,0))))
    out_grad_ = out_grad.permute(1, 2, 0, 3) # (NH, NW, N, OC) => (K', K', IC', OC)
    W_grad = torch.nn.functional.conv2d(Z0.permute(0, 3, 1, 2), out_grad_.permute(3, 2, 0, 1))
    W_grad = W_grad.permute(0, 2, 3, 1).permute(1, 2, 0, 3)
    return Z_grad.detach().numpy(), W_grad.detach().numpy()

if __name__ == '__main__':
    if random:
        Z_ = np.random.randn(N, H, W, IC)
        W_ = np.random.randn(K, K, IC, OC)
    else:
        Z_ = np.arange(1, N*H*W*IC+1).reshape((N, H, W, IC))
        W_ = np.ones((K, K, IC, OC))

    Z_ = torch.Tensor(Z_)
    W_ = torch.Tensor(W_)
    Z_.requires_grad=True
    W_.requires_grad=True

    torch_Z_grad, torch_W_grad = torch_conv_backward(Z_, W_)
    # print(f'torch_Z_grad: {torch_Z_grad.shape} \n\n {torch_Z_grad}')
    # print(f'torch_W_grad: {torch_W_grad.shape} \n\n {torch_W_grad}')
    out_grad = torch.Tensor(np.ones((N, NH, NW, OC)))
    Z_grad, W_grad = manual_conv_backward(Z_, W_, out_grad)
    np.testing.assert_allclose(torch_Z_grad, Z_grad, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(torch_W_grad, W_grad, rtol=1e-5, atol=1e-5)
