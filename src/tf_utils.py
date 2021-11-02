import tensorflow as tf
import numpy as np


def _maybe_validate_matrix(a, validate_args):
    """Checks that input is a `float` matrix."""
    assertions = []
    if not a.dtype.is_floating:
        raise TypeError("Input `a` must have `float`-like `dtype` " "(saw {}).".format(a.dtype.name))
    if a.shape.ndims is not None:
        if a.shape.ndims < 2:
            raise ValueError("Input `a` must have at least 2 dimensions " "(saw: {}).".format(a.shape.ndims))
    elif validate_args:
        assertions.append(
            tf.compat.v1.assert_rank_at_least(a, rank=2, message="Input `a` must have at least 2 dimensions.")
        )
    return assertions


def pinv_naive(a):
    """Returns the Moore-Penrose pseudo-inverse"""
    # dtype = a.dtype.as_numpy_dtype
    # S, U, V = tf.linalg.svd(a, full_matrices=True, compute_uv=True)
    # S = tf.expand_dims(S, 1)
    #
    # # calculate pseudo inverse via SVD
    # # not good, will produce inf when divide by 0
    # threshold = tf.reduce_max(S) * 1e-5
    # S = tf.where(S > threshold, S, tf.fill(tf.shape(input=S), np.array(np.inf, dtype)))
    # a_pinv = tf.matmul(V/S, tf.transpose(U))
    # return a_pinv
    s, u, v = tf.linalg.svd(a)

    threshold = tf.reduce_max(s) * 1e-5
    s_mask = tf.boolean_mask(s, s > threshold)  # s[s>threshold]
    s_inv = tf.linalg.diag(tf.concat([1.0 / s_mask, tf.zeros([tf.size(s) - tf.size(s_mask)])], 0))

    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))


def pinv(a, rcond=None, validate_args=False, name=None):
    """
    https://github.com/tensorflow/probability/blob/d674d79bc8175bff2f415bf3b38a42f51ffc999c/tensorflow_probability/python/math/linalg.py
    Compute the Moore-Penrose pseudo-inverse of a matrix.
    Calculate the [generalized inverse of a matrix](
    https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) using its
    singular-value decomposition (SVD) and including all large singular values.
    The pseudo-inverse of a matrix `A`, is defined as: "the matrix that 'solves'
    [the least-squares problem] `A @ x = b`," i.e., if `x_hat` is a solution, then
    `A_pinv` is the matrix such that `x_hat = A_pinv @ b`. It can be shown that if
    `U @ Sigma @ V.T = A` is the singular value decomposition of `A`, then
    `A_pinv = V @ inv(Sigma) U^T`. [(Strang, 1980)][1]
    This function is analogous to [`numpy.linalg.pinv`](
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html).
    It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
    default `rcond` is `1e-15`. Here the default is
    `10. * max(num_rows, num_cols) * np.finfo(dtype).eps`.
    Args:
      a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
        pseudo-inverted.
      rcond: `Tensor` of small singular value cutoffs.  Singular values smaller
        (in modulus) than `rcond` * largest_singular_value (again, in modulus) are
        set to zero. Must broadcast against `tf.shape(a)[:-2]`.
        Default value: `10. * max(num_rows, num_cols) * np.finfo(a.dtype).eps`.
      validate_args: When `True`, additional assertions might be embedded in the
        graph.
        Default value: `False` (i.e., no graph assertions are added).
      name: Python `str` prefixed to ops created by this function.
        Default value: "pinv".
    Returns:
      a_pinv: The pseudo-inverse of input `a`. Has same shape as `a` except
        rightmost two dimensions are transposed.
    Raises:
      TypeError: if input `a` does not have `float`-like `dtype`.
      ValueError: if input `a` has fewer than 2 dimensions.
    #### Examples
    ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    a = tf.constant([[1.,  0.4,  0.5],
                     [0.4, 0.2,  0.25],
                     [0.5, 0.25, 0.35]])
    tf.matmul(tfp.math.pinv(a), a)
    # ==> array([[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]], dtype=float32)
    a = tf.constant([[1.,  0.4,  0.5,  1.],
                     [0.4, 0.2,  0.25, 2.],
                     [0.5, 0.25, 0.35, 3.]])
    tf.matmul(tfp.math.pinv(a), a)
    # ==> array([[ 0.76,  0.37,  0.21, -0.02],
                 [ 0.37,  0.43, -0.33,  0.02],
                 [ 0.21, -0.33,  0.81,  0.01],
                 [-0.02,  0.02,  0.01,  1.  ]], dtype=float32)
    ```
    #### References
    [1]: G. Strang. "Linear Algebra and Its Applications, 2nd Ed." Academic Press,
         Inc., 1980, pp. 139-142.
    """
    with tf.compat.v1.name_scope(name, "pinv", [a, rcond]):
        a = tf.convert_to_tensor(value=a, name="a")

        assertions = _maybe_validate_matrix(a, validate_args)
        if assertions:
            with tf.control_dependencies(assertions):
                a = tf.identity(a)

        dtype = a.dtype.as_numpy_dtype

        if rcond is None:

            def get_dim_size(dim):
                if tf.compat.dimension_value(a.shape[dim]) is not None:
                    return tf.compat.dimension_value(a.shape[dim])
                return tf.shape(input=a)[dim]

            num_rows = get_dim_size(-2)
            num_cols = get_dim_size(-1)
            if isinstance(num_rows, int) and isinstance(num_cols, int):
                max_rows_cols = float(max(num_rows, num_cols))
            else:
                max_rows_cols = tf.cast(tf.maximum(num_rows, num_cols), dtype)
            rcond = 10.0 * max_rows_cols * np.finfo(dtype).eps

        rcond = tf.convert_to_tensor(value=rcond, dtype=dtype, name="rcond")

        # Calculate pseudo inverse via SVD.
        # Note: if a is symmetric then u == v. (We might observe additional
        # performance by explicitly setting `v = u` in such cases.)
        [
            singular_values,  # Sigma
            left_singular_vectors,  # U
            right_singular_vectors,  # V
        ] = tf.linalg.svd(a, full_matrices=False, compute_uv=True)

        # Saturate small singular values to inf. This has the effect of make
        # `1. / s = 0.` while not resulting in `NaN` gradients.
        cutoff = rcond * tf.reduce_max(input_tensor=singular_values, axis=-1)
        singular_values = tf.where(
            singular_values > cutoff[..., tf.newaxis],
            singular_values,
            tf.fill(tf.shape(input=singular_values), np.array(np.inf, dtype)),
        )

        # Although `a == tf.matmul(u, s * v, transpose_b=True)` we swap
        # `u` and `v` here so that `tf.matmul(pinv(A), A) = tf.eye()`, i.e.,
        # a matrix inverse has "transposed" semantics.
        a_pinv = tf.matmul(
            right_singular_vectors / singular_values[..., tf.newaxis, :],
            left_singular_vectors,
            adjoint_b=True,
        )

        if a.shape.ndims is not None:
            a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

        return a_pinv
