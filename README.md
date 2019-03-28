# IRLS in tensorflow2.0/pytorch
* IRLS(Iterative re-weighted least square)  for Logistic Regression,
implemented using tensorflow2.0 and pytorch (tensorflow < 2.0 is also supported).

* Note that IRLS is a second order optimization problem, which is equivalent to Newton's method.

* We show that both tensorflow and pytorch can do general matrix based algorithms, and
both can be accelerated by the power of gpu.

* In this implementation, we use svd to solve pseudo inverse of singular matrices. Note that
the current pseudo inverse algorithm can have precision issue. A more robust one like the
ones in numpy or pytorch could potentially improve.
