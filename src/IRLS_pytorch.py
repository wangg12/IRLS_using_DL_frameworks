# python 3
from __future__ import print_function, division, absolute_import

import os
import argparse
import random
import numpy as np

# from numpy import linalg
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

# from scipy.sparse import linalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


path_train = "../a9a/a9a"
path_test = "../a9a/a9a.t"
MAX_ITER = 100
np_dtype = np.float32

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


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
    res = torch.mm(Xw.t(), y) - torch.log(1 + Xw.exp()).sum()
    if L2_param != None and L2_param > 0:
        res += -0.5 * L2_param * torch.mm(w.t(), w)
    return -res


def prob(X, w):
    """
  X: Nxd
  w: dx1
  ---
  prob: N x num_classes(2)"""
    Xw = X.mm(w)
    y = torch.tensor([[0.0, 1.0]], device=device)  # 1x2
    return (Xw * y).exp() / (1 + Xw.exp())  # Nx2


def compute_acc(X, y, w):
    p = prob(X, w)
    y_pred = torch.argmax(p, 1).to(y)
    return (y.flatten() == y_pred).float().mean()


def pinv(A):
    """
    https://discuss.pytorch.org/t/torch-pinverse-seems-to-be-inaccurate/33616
    Return the pseudoinverse of A,
    without invoking the SVD in torch.pinverse().

    Could also use (but doesn't avoid the SVD):
        R.pinverse().matmul(Q.t())
    """
    rows,cols = A.size()
    if rows >= cols:
        Q,R = torch.qr(A)
        return R.inverse().mm(Q.t())
    else:
        Q,R = torch.qr(A.t())
        return R.inverse().mm(Q.t()).t()


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

    XRX = torch.mm(X.t(), R_flat.expand_as(X) * X)  # dxd
    if L2_param > 0:
        XRX.diagonal().add_(L2_param)

    np.save('XRX_pytorch.npy', XRX.cpu().numpy())

    # Calculate pseudo inverse via SVD
    # For singular matrices, we only invert the non-zero singular values.
    # not really stable, pytorch/numpy style pinverse, which invert the singular
    # values above certain threshold (computed with the max singular value)
    # should improve this. But this is here to match tf.
    U, S, V = torch.svd(XRX, some=False)
    S_pinv = torch.where(S != 0, 1/S, torch.zeros_like(S))
    XRX_pinv = torch.chain_matmul(V, S_pinv.diag(), U.t())

    # w = w - (X^T R X)^(-1) X^T (mu-y)
    val = torch.mm(X.t(), mu - y)
    if L2_param > 0:
        val += L2_param * w_old

    w_update = torch.mm(XRX_pinv, val)
    w_new = w_old - w_update
    return w_new


def train_IRLS(
    X_train, y_train, X_test=None, y_test=None, L2_param=0, max_iter=MAX_ITER
):
    """train Logistic Regression via IRLS algorithm
  X: Nxd
  y: Nx1
  ---

  """
    N, d = X_train.shape
    X_train = torch.as_tensor(X_train, device=device)
    X_test = torch.as_tensor(X_test, device=device)
    y_train = torch.as_tensor(y_train, device=device)
    y_test = torch.as_tensor(y_test, device=device)

    w = torch.full((d, 1), 0.01, device=device)

    print("start training...")
    print("Device: {}".format(device))
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

        L2_norm_w = torch.norm(w)
        print("\t L2 norm of w: {}".format(L2_norm_w.item()))

        if i > 0:
            diff_w = torch.norm(w - w_old_data)
            print("\t diff of w_old and w: {}".format(diff_w.item()))
            if diff_w < 1e-2:
                break

        w_old_data = w.clone()
        w = update_weight(w, X_train, y_train, L2_param)
        i += 1
    print("training done.")


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
