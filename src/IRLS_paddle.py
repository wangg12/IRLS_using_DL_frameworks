# python 3
from __future__ import print_function, division, absolute_import

import os.path as osp
import argparse
import random
from loguru import logger
import numpy as np
import time

# from numpy import linalg
import paddle


from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

# from scipy.sparse import linalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cur_dir = osp.dirname(osp.abspath(__file__))
path_train = osp.join(cur_dir, "../a9a/a9a")
path_test = osp.join(cur_dir, "../a9a/a9a.t")
MAX_ITER = 100
np_dtype = np.float32

paddle.device.set_device("gpu")

# manual seed
manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)

# load all data
X_train, y_train = load_svmlight_file(path_train, n_features=123, dtype=np_dtype)
X_test, y_test = load_svmlight_file(path_test, n_features=123, dtype=np_dtype)
# X: scipy.sparse.csr.csr_matrix

# X_train: (32561, 123), y_train: (32561,)
# X_test:  (16281, 123), y_test:(16281,)

# stack a dimension of ones to X to simplify computation
N_train = X_train.shape[0]
N_test = X_test.shape[0]
X_train = np.hstack((np.ones((N_train, 1), dtype=np_dtype), X_train.toarray()))
X_test = np.hstack((np.ones((N_test, 1), dtype=np_dtype), X_test.toarray()))

# X_train = csr_matrix(X_train, dtype=dtype)
# X_test = csr_matrix(X_test, dtype=dtype)

y_train = y_train.reshape((N_train, 1))
y_test = y_test.reshape((N_test, 1))

# label: -1, +1 ==> 0, 1
y_train = np.float32(np.where(y_train == -1, 0, 1))
y_test = np.float32(np.where(y_test == -1, 0, 1))


# NB: here X's shape is (N,d), which differs to the derivation


def neg_log_likelihood(w, X, y, L2_param=None):
    """
    w: dx1
    X: Nxd
    y: Nx1
    L2_param: \lambda>0, will introduce -\lambda/2 ||w||_2^2
    """
    Xw = X.mm(w)
    res = paddle.mm(Xw.t(), y) - paddle.log(1 + Xw.exp()).sum()
    if L2_param != None and L2_param > 0:
        res += -0.5 * L2_param * paddle.mm(w.t(), w)
    return -res


def prob(X, w):
    """
    X: Nxd
    w: dx1
    ---
    prob: N x num_classes(2)"""
    Xw = X.mm(w)
    y = paddle.to_tensor([[0.0, 1.0]])  # 1x2
    return (Xw * y).exp() / (1 + Xw.exp())  # Nx2


def compute_acc(X, y, w):
    p = prob(X, w)
    y_pred = paddle.argmax(p, 1)
    return (y.flatten() == y_pred).to("float32").mean()


def pinv_naive(A):
    dtype = A.dtype
    U, S, Vh = paddle.linalg.svd(A, full_matrices=True)
    threshold = paddle.max(S) * 1e-5
    S_mask = S[S > threshold]
    S_pinv = paddle.concat([1.0 / S_mask, paddle.full([S.numel() - S_mask.numel()], 0.0, dtype=dtype)], 0)
    # A_pinv = V @ S_pinv.diag() @ U.t()
    A_pinv = Vh.t() @ S_pinv.diag() @ U.t()
    return A_pinv


def update_weight(w_old, X, y, L2_param=0):
    """
    w_new = w_old - w_update
    w_update = (X'RX+lambda*I)^(-1) (X'(mu-y) + lambda*w_old)
    lambda is L2_param

    w_old: dx1
    X: Nxd
    y: Nx1
    ---
    w_new: dx1
    """
    mu = X.mm(w_old).sigmoid()  # Nx1

    R_flat = mu * (1 - mu)  # element-wise, Nx1

    XRX = paddle.mm(X.t(), R_flat.expand_as(X) * X)  # dxd
    if L2_param > 0:
        # XRX.diagonal().add_(L2_param)
        for i in range(XRX.shape[0]):
            XRX[i, i] += L2_param

    # np.save('XRX_paddle.npy', XRX.cpu().numpy())

    # Method 1: Calculate pseudo inverse via SVD
    # For singular matrices, we invert the singular
    # values above certain threshold (computed with the max singular value)
    # this is slightly better than paddle.linalg.pinv when L2_param=0
    XRX_pinv = pinv_naive(XRX)

    # method 2
    # XRX_pinv = paddle.linalg.pinv(XRX)

    # w = w - (X^T R X)^(-1) X^T (mu-y)
    val = paddle.mm(X.t(), mu - y)
    if L2_param > 0:
        val += L2_param * w_old

    w_update = paddle.mm(XRX_pinv, val)
    w_new = w_old - w_update
    return w_new


@logger.catch
def train_IRLS(X_train, y_train, X_test=None, y_test=None, L2_param=0, max_iter=MAX_ITER):
    """train Logistic Regression via IRLS algorithm
    X: Nxd
    y: Nx1
    ---

    """
    N, d = X_train.shape
    X_train = paddle.to_tensor(X_train)
    X_test = paddle.to_tensor(X_test)
    y_train = paddle.to_tensor(y_train)
    y_test = paddle.to_tensor(y_test)

    w = paddle.full((d, 1), 0.01)

    print("start training...")
    tic = time.time()
    print("Device: {}".format(paddle.device.get_device()))
    print("L2 param(lambda): {}".format(L2_param))
    i = 0
    # iteration
    while i <= max_iter:
        print("iter: {}".format(i))

        neg_L = neg_log_likelihood(w, X_train, y_train, L2_param)
        print("\t neg log likelihood: {}".format(neg_L.sum()))

        train_acc = compute_acc(X_train, y_train, w)
        test_acc = compute_acc(X_test, y_test, w)
        print("\t train acc: {}, test acc: {}".format(train_acc, test_acc))

        L2_norm_w = paddle.norm(w)
        print("\t L2 norm of w: {}".format(L2_norm_w.item()))

        if i > 0:
            diff_w = paddle.norm(w - w_old_data)
            print("\t diff of w_old and w: {}".format(diff_w.item()))
            if diff_w < 1e-2:
                break

        w_old_data = w.clone()
        w = update_weight(w, X_train, y_train, L2_param)
        i += 1
    print(f"training done, using {time.time() - tic}s.")


if __name__ == "__main__":
    lambda_ = 20  # 0
    train_IRLS(X_train, y_train, X_test, y_test, L2_param=lambda_, max_iter=100)

    # from sklearn.linear_model import LogisticRegression
    # classifier = LogisticRegression()
    # classifier.fit(X_train, y_train.reshape(N_train,))
    # y_pred_train = classifier.predict(X_train)
    # train_acc = np.sum(y_train.reshape(N_train,) == y_pred_train)/N_train
    # print('train_acc: {}'.format(train_acc))
    # y_pred_test = classifier.predict(X_test)
    # test_acc = np.sum(y_test.reshape(N_test,) == y_pred_test)/N_test
    # print('test acc: {}'.format(test_acc))
