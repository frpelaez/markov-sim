import numpy as np


def matpow(x: np.ndarray, k: int) -> np.ndarray:
    if x.shape[0] != x.shape[1]:
        raise ValueError("x must be a square matrix")
    if k < 0:
        raise ValueError("k must be a non ngeative integer")
    elif k == 0:
        return np.eye(x.shape[0])
    elif k == 1:
        return x
    elif k % 2 == 0:
        rec = matpow(x, k // 2)
        return np.matmul(rec, rec)
    else:
        rec = matpow(x, k // 2)
        res = np.matmul(rec, rec)
        return np.matmul(rec, res)
