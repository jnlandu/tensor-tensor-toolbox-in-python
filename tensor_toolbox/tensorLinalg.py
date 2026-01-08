"""
Docstring for tensor_toolbox.tensorLinalg
"""

# In this file, we define various operation  of tensor algebra under the t-product.
# For the theory  behind, we recommend the seminal paper:
# "Third-Order Tensors as Operators on Matrices: A Theoretical and Computational Framework
# for Tensor Decompositions" by Kilmer and Martin, 2011.

#  A Matlab implementation of these operations can be found at:
#  https://csmr.ca.sandia.gov/~tgkolda/TensorToolbox/



from tensor_toolbox.config import *

def bcirc(AA):
  """
  Convert a 3D tensor into a block circulant matrix,
  See the definition of circulant matrix in https://en.wikipedia.org/wiki/Circulant_matrix
  See the paper Kilmer et al for t-tensor linear algebra

  Parameters
  ----------
  AA: torch. Tensor of size (mm,nn, pp)

  Returns:
  --------
  BB: torch. Tensor of size (mm*nn, mm*pp)
  """
  mm, nn, _ = AA.shape
  
  #  Stack all the 3-mode slices vertically:
  BB = BB = torch.cat([AA[:, :, k] for k in range(nn)], dim=0)

  #  Then create the bcirc structure
  newCol = BB.clone()
  for  _ in range(1, nn):
    newCol = torch.roll(newCol, mm, dims =0)
    BB = torch.cat([BB, newCol], dim=1)

  return BB

# 1. Block Circulant Matrix from a 3rd-order Tensor
# def bcirc(tensor):
#     """
#     Constructs the block circulant matrix from a 3rd-order tensor.

#     Parameters:
#     tensor (ndarray): A 3rd-order tensor of shape (n1, n2, n3).

#     Returns:
#     ndarray: The block circulant matrix of shape (n1*n3, n2*n3).
#     """
#     n1, n2, n3 = tensor.shape
#     bcirc_matrix = np.zeros((n1 * n3, n2 * n3), dtype=tensor.dtype)

#     for i in range(n3):
#         for j in range(n3):
#             block = tensor[:, :, (i - j) % n3]
#             bcirc_matrix[i * n1:(i + 1) * n1, j * n2:(j + 1) * n2] = block

#     return bcirc_matrix

# 2. T-Product of two 3rd-order Tensors
def t_product(A, B, device = device):
  """
  Tensor-tensor produdct (t-product) using FFT mode-3, as defined in the Kilmer et al paper.

  Parameters:
  -----------
  A: torch.Tensor of size (n1, n2, n3)
  B: torch.Tensor of size (n2, n4,n3)

  Returns:
  --------
  C: torch.Tensor of size (n1, n4, n3)
  """
  _, n2, n3 = A.shape
  na, _, nb = B.shape

  # Check if the dimensions are compatible
  if n2 !=  na or n3 != nb:
    raise ValueError("Dimensions of A and B are not compatible for t-product. Need A.shape =(n1,n2,n3) et B.shape =(n2, n4, n3)")
  
  #  Applique FFT le long de mode 3
  Af = torch.fft.fftn(A, dim=2).to(device)
  Bf = torch.fft.fftn(B, dim=2).to(device)

  # Batched matrix multiply le long the slices de frequence
  #  (n3, n1, n2) @ (n3, n2, n4) -> (n3, n1, n4)
  Cf = torch.matmul(Af.permute(2, 0, 1), Bf.permute(2, 0, 1)).permute(1, 2, 0).to(device)

  #  Inverse FFT
  C = torch.fft.ifftn(Cf, dim=2).to(device)

  if (not torch.is_complex(A)) and (not torch.is_complex(B)):
    return C.real

  return C


# 3. T-Transpose of a 3rd-order Tensor

def t_transpose(A):
  """
  Tensor transpose of A, as defined in the Kilmer et al paper.

  Parameters:
  -----------
  A: torch.Tensor of size (n1, n2, n3)

  Returns:
  --------
  B: torch.Tensor of size (n2, n1, n3)
  """
  # print("Debug t_trans A shape:", A.shape)
  n3 = A.shape[2]

  # Create index array [0,n3-1, n3-2, ..., 1]
  idx = [0] + list(range(n3 - 1, 0, -1))

  #  Index le long  dim 3

  Z = A[:, :, idx]

  # print("Debug t_trans Z:", Z.shape)

  #  Permute dimensions: swap first and second dimensions
  Z = Z.permute(1, 0, 2)

  return Z

# 4. Unfold and Fold Operators

def unfold(B):
  """
  Unfold operatio  frpm the  Kilmer et al paper.
  Unfold stacks frontal slices vertically to obtain a matrix.
  Parameters:
  -----------
  B: torch.Tensor of size (n1, n2, n3)

  Returns:
  --------
  C: torch.Tensor of size (n1*n2, n3)
  """
  _, _, n3 = B.shape
  return torch.cat([B[:, :, k] for k in range(n3)], dim=0)

def fold(A):
  """
  Fold operator from the Kilmer et al paper.
  Fold is the inverse of Unfold. It maps a matrix to a tensor.
  Parameters:
  -----------
  A: torch.Tensor of size (n1*n2, n3)

  Returns:
  --------
  B: torch.Tensor of size (n1, n2, n3)

  """
  n1, _, n3 = A.shape
  # return A.reshape(n1, n2, n3)
  return torch.stack([A[kk*n1:(kk+1)*n1, :] for kk in range(n3)], dim=2)

# 5. Identity Tensor
def identity_tensor(n, m, device = device):
  """
  Create an identity tensor of size (n, n, m)
  Parameters:
  -----------
  n: int, number of rows and columns
  m: int, number of frontal slices

  Returns:
  --------
  I: torch.Tensor of size (n, n, m)
  """
  I = torch.zeros((n, n, m), device = device)
  for k in range(m):
    I[:, :, k] = torch.eye(n, device = device)
  return I


def t_frobenius_norm(A):
    """
    Compute Frobenius norm of tensor.

    Parameters:
    -----------
    A : torch. 3rd order tensor A(m,n, p)
    Returns:
    --------
    frob : torch. scalar. Frobenius norm of A.
    """
    frob = torch.linalg.norm(A)
    return  frob 


def tube_tpinv(G, tol=1e-12):
    """
    T-product pseudoinverse of a tube tensor G of shape (1,1,p) under t-product.
    Computed slice-wise in FFT domain: pinv of scalars.
    
    Parameters:
    -----------
    G: torch. Tensor of size (1,1,p)
    
    Returns:
    --------
    Ginv: torch. Tensor of size (1,1,p)
    """
    assert G.shape[0] == 1 and G.shape[1] == 1, "G must be (1,1,p)"

    Gf = torch.fft.fft(G, dim=2)                 # (1,1,p) complex

    mag = Gf.abs()

    Gf_inv = torch.where(mag > tol, 1.0 / Gf, torch.zeros_like(Gf))

    Ginv = torch.fft.ifft(Gf_inv, dim=2)

    # If G was real, keep real (numerical FFT noise)
    if (not torch.is_complex(G)):
        Ginv = Ginv.real
        
    return Ginv


def t_pinv_apply(Ablk, Bblk, rcond=1e-12):
    """
    Apply t-pseudoinverse: X = (Ablk)^dagger * Bblk   under t-product.
    Implemented via FFT along mode-3, then slice-by-slice matrix pinv.

    Ablk: (m, r, p)
    Bblk: (m, k, p)
    Returns:
        X: (r, k, p)
    """
    # FFT along 3rd dim
    Af = torch.fft.fft(Ablk, dim=2)  # (m, r, p) complex
    Bf = torch.fft.fft(Bblk, dim=2)  # (m, k, p) complex

    _, r, p = Ablk.shape
    _, k, _ = Bblk.shape

    Xf = torch.empty((r, k, p), device=Ablk.device, dtype=Af.dtype)

    # Solve each frontal slice: Xf(:,:,s) = pinv(Af(:,:,s)) @ Bf(:,:,s)
    for s in range(p):
        As = Af[:, :, s]  # (m, r)
        Bs = Bf[:, :, s]  # (m, k)
        Xf[:, :, s] = torch.linalg.pinv(As, rcond=rcond) @ Bs

    # Back to spatial domain (real tensors assumed)
    X = torch.fft.ifft(Xf, dim=2).real
    return X



