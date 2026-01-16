


import needle as ndl


def tril(tensor: ndl.Tensor) -> ndl.Tensor:
    if len(tensor.shape) != 2:
        raise ValueError("Only support 2D tensor")

    rows, cols = tensor.shape
    col_idx = ndl.arange(cols, device=tensor.device)
    row_idx = ndl.arange(rows, device=tensor.device).unsqueeze(1)
    mask = col_idx <= row_idx

    return tensor * mask

def block_diag(*tensors: ndl.Tensor) -> ndl.Tensor:
    tensors = list(tensors)
    total_rows = sum(tensor.shape[0] for tensor in tensors)
    total_cols = sum(tensor.shape[1] for tensor in tensors)
    device = tensors[0].device
    dtype = tensors[0].dtype
    result = ndl.zeros(total_rows, total_cols, device=device, dtype=dtype)
    row_st = 0
    col_st = 0
    for tensor in tensors:
        row, col = tensor.shape
        result[row_st : row_st + row, col_st : col_st + col] = tensor
        row_st += row
        col_st += col
    return result