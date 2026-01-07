## Tensor-Tensor Toolbox in Python

This repo povides the python implementation of basic tensor operations under t-product defined in the paper by [J. M. Kilmer and C. D. Martin, "Factorization strategies for third-order tensors," Linear Algebra and its Applications, vol. 435, no. 3, pp. 641-658, Aug. 2011](https://www.sciencedirect.com/science/article/pii/S0024379510004830).

The implementations follows from the repo [Tensor-tensor-product-toolbox](https://github.com/canyilu/Tensor-tensor-product-toolbox), which presents a MATLAB implementation of the same operations.

To get you understand the t-product and its applications, please refer to the following papers:
- [J. M. Kilmer, K. Braman, N. Hao, and R. C. Hoover, "Third-order tensors as operators on matrices: a theoretical and computational framework with applications in imaging," SIAM Journal on Matrix Analysis and Applications, vol. 34, no. 1, pp. 148-172, Jan. 2013.](https://epubs.siam.org/doi/10.1137/120876800)
- [C. Lu, J. Feng, Y. Chen, W. Liu, Z. Lin, and S. Yan, "Tensor robust principal component analysis with a new tensor nuclear norm," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 925-938, Apr. 2020.](https://ieeexplore.ieee.org/document/8355062)


**Key ideas of t-product:** Extends faithfully  matrix linear algebra to  3-order tensors by defining a new multiplication operation, called t-product, between two 3-order tensors. The t-product is defined based on the circular convolution between the tubes of two tensors. Many matrix concepts such as transpose, identity, inverse, orthogonality, and SVD can be extended to tensors under the t-product framework.

## Quick start
```python
pip install tensor-toolbox
```
or from the repository
```python
python -m pip install -q git+https://github.com/jnlandu/tensor-tensor-toolbox-in-python.git
```

```python
from tensor_toolbox.CONFIG import *
from tensor_toolbox.tensorLinalg import *
```

## Usage

### Device and dtype

`tensor_toolbox.CONFIG` defines:
- `device`: selected automatically (Apple MPS if available, else CUDA, else CPU)
- `dtype`: default tensor dtype (currently `torch.float32`)

Typical pattern:

```python
import torch

from tensor_toolbox.CONFIG import device, dtype
from tensor_toolbox.tensorLinalg import (
	bcirc,
	fold,
	identity_tensor,
	t_frobenius_norm,
	t_pinv_apply,
	t_product,
	t_transpose,
	unfold,
)

torch.manual_seed(0)
```

### Create tensors

All core ops assume 3rd-order tensors of shape `(n1, n2, n3)`.

```python
A = torch.randn(2, 3, 4, device=device, dtype=dtype)
B = torch.randn(3, 5, 4, device=device, dtype=dtype)
```

### t-product

```python
C = t_product(A, B)
print(C.shape)  # (2, 5, 4)
```

### Identity tensor (under t-product)

```python
I = identity_tensor(n=A.shape[1], m=A.shape[2], device=device)
err = t_frobenius_norm(t_product(I, A) - A).item()
print("||I*A - A||_F =", err)
```

### t-transpose

```python
At = t_transpose(A)
print(At.shape)  # (3, 2, 4)
```

### unfold / fold

```python
U = unfold(A)
A2 = fold(U)
print(t_frobenius_norm(A2 - A).item())
```

### bcirc

```python
BA = bcirc(A)
print(BA.shape)
```

### t-pseudoinverse apply (small demo)

Solve (approximately) `Ablk * X ≈ Bblk` under the t-product, via FFT + slice-wise pseudoinverse.

```python
Ablk = torch.randn(4, 2, 3, device=device, dtype=dtype)  # (m, r, p)
Bblk = torch.randn(4, 1, 3, device=device, dtype=dtype)  # (m, k, p)

X = t_pinv_apply(Ablk, Bblk)
R = t_product(Ablk, X) - Bblk
print("||A*X - B||_F =", t_frobenius_norm(R).item())
```


## Requirements & Installation
#### Requirements
```text
- Python 3.8+
- PyTorch 1.10+
- numpy 
```
#### Install
You can  clone the repo and install it in editable mode, or install directly from GitHub.

Clone the repository:
```bash
git clone https://github.com/jnlandu/tensor-tensor-toolbox-in-python
cd tensor-tensor-toolbox-in-python
``` 

Install in editable mode:
```bash
python -m pip install -e .
```

or fron install directly from GitHub:

```bash
python -m pip install "git+https://github.com/jnlandu/tensor-tensor-toolbox-in-python.git"
```

If you have made changes in your copy of the repository and want to update the installed package, run:

```bash
python -m pip install -e . --upgrade
```
You can build the source distribution and install it:

```bash
python -m build --sdist
python -m pip install dist/tensor_toolbox-0.1.0.tar.gz
```
or build and install a wheel:
```bash
python -m build
python -m pip install dist/tensor_toolbox-0.1.0-py3-none-any.whl
```

## Folder structure
- `tensor_toolbox/`: Contains the implementation of tensor operations.
- `examples.ipynb`: A Jupyter notebook with examples.   

### Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.





