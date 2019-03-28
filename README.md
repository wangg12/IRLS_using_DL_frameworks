# IRLS in tensorflow2.0/pytorch
* IRLS(Iterative re-weighted least square)  for Logistic Regression,
implemented using tensorflow2.0 and pytorch (tensorflow < 2.0 is also supported).

* Note that IRLS is a second order optimization problem, which is equivalent to Newton's method.

* We show that both tensorflow and pytorch can do general matrix based algorithms, and
both can be accelerated by the power of gpu.

* In this implementation, we use svd to solve pseudo inverse of singular matrices.
The svd in torch is somehow less numerically stable than that in tensorflow, so we
use a slight regularization in the vanilla IRLS in pytorch.

The performance of pytorch version is much lower than that of the tensorflow(2.0) version, perhaps it is due to the numeric stability of `torch.svd`.
