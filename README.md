# IRLS in tensorflow2.0/pytorch
* IRLS(Iterative re-weighted least square) for Logistic Regression, implemented using
  * tensorflow < 2.0
  * tensorflow2.0
  * pytorch
  * megengine

* Note that IRLS is a second order optimization problem, which is equivalent to Newton's method.

* We show that these DL frameworks can do general matrix based algorithms, and can be accelerated by the power of gpu.

* In this implementation, we use svd to solve pseudo inverse of singular matrices.
