## Tensor - Tensor toolbox in Python

This repo povides the python implementation of basic tensor operations under t-product defined in the paper by [J. M. Kilmer and C. D. Martin, "Factorization strategies for third-order tensors," Linear Algebra and its Applications, vol. 435, no. 3, pp. 641-658, Aug. 2011](https://www.sciencedirect.com/science/article/pii/S0024379510004830).

The implementations follows from the repo [Tensor-tensor-product-toolbox](https://github.com/canyilu/Tensor-tensor-product-toolbox), which presents a MATLAB implementation of the same operations.

To get you understand the t-product and its applications, please refer to the following papers:
- [J. M. Kilmer, K. Braman, N. Hao, and R. C. Hoover, "Third-order tensors as operators on matrices: a theoretical and computational framework with applications in imaging," SIAM Journal on Matrix Analysis and Applications, vol. 34, no. 1, pp. 148-172, Jan. 2013.](https://epubs.siam.org/doi/10.1137/120876800)
- [C. Lu, J. Feng, Y. Chen, W. Liu, Z. Lin, and S. Yan, "Tensor robust principal component analysis with a new tensor nuclear norm," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 925-938, Apr. 2020.](https://ieeexplore.ieee.org/document/8355062)


**Key ideas of t-product:** Extends faithfully  matrix linear algebra to  3-order tensors by defining a new multiplication operation, called t-product, between two 3-order tensors. The t-product is defined based on the circular convolution between the tubes of two tensors. Many matrix concepts such as transpose, identity, inverse, orthogonality, and SVD can be extended to tensors under the t-product framework.

## Requirements
- Python 3.x
- Numpy
- torch (optional, for GPU support)

## Installation
You can clone this repository using git:
```bash
git clone   
```

### Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.





