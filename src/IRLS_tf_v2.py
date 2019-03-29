# python 3
# tensorflow 2.0
from __future__ import print_function, division, absolute_import

import os
import argparse
import random
import numpy as np
import datetime

# from numpy import linalg
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(1, osp.join(cur_dir, '.'))
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

# from scipy.sparse import linalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_utils import pinv_naive, pinv


path_train = osp.join(cur_dir, "../a9a/a9a")
path_test = osp.join(cur_dir, "../a9a/a9a.t")
MAX_ITER = 100
np_dtype = np.float32
tf_dtype = tf.float32

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
X_train = np.hstack((np.ones((N_train, 1)), X_train.toarray())).astype(np_dtype)
X_test = np.hstack((np.ones((N_test, 1)), X_test.toarray())).astype(np_dtype)
# print(X_train.shape, X_test.shape)

y_train = y_train.reshape((N_train, 1))
y_test = y_test.reshape((N_test, 1))

# label: -1, +1 ==> 0, 1
y_train = np.where(y_train == -1, 0, 1)
y_test = np.where(y_test == -1, 0, 1)

# NB: here X's shape is (N,d), which differs to the derivation

def neg_log_likelihood(w, X, y, L2_param=None):
    """
    w: dx1
    X: Nxd
    y: Nx1
    L2_param: \lambda>0, will introduce -\lambda/2 ||w||_2^2
    """
    # print(type(X), X.dtype)
    res = tf.matmul(tf.matmul(tf.transpose(w), tf.transpose(X)), y.astype(np_dtype)) - \
            tf.reduce_sum(tf.math.log(1 + tf.exp(tf.matmul(X, w))))
    if L2_param != None and L2_param > 0:
        res += -0.5 * L2_param * tf.matmul(tf.transpose(w), w)
    return -res[0][0]


def prob(X, w):
    """
    X: Nxd
    w: dx1
    ---
    prob: N x num_classes(2)"""
    y = tf.constant(np.array([0.0, 1.0]), dtype=tf.float32)
    prob = tf.exp(tf.matmul(X, w) * y) / (1 + tf.exp(tf.matmul(X, w)))
    return prob


def compute_acc(X, y, w):
    p = prob(X, w)
    y_pred = tf.cast(tf.argmax(p, axis=1), tf.float32)
    y = tf.cast(tf.squeeze(y), tf.float32)
    acc = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
    return acc


def update(w_old, X, y, L2_param=0):
    """
  w_new = w_old - w_update
  w_update = (X'RX+lambda*I)^(-1) (X'(mu-y) + lambda*w_old)
  lambda is L2_param

  w_old: dx1
  X: Nxd
  y: Nx1
  ---
  w_update: dx1
  """
    d = X.shape[1]
    mu = tf.sigmoid(tf.matmul(X, w_old))  # Nx1

    R_flat = mu * (1 - mu)  # element-wise, Nx1

    L2_reg_term = L2_param * tf.eye(d)
    XRX = tf.matmul(tf.transpose(X), R_flat * X) + L2_reg_term  # dxd
    # np.save('XRX_tf.npy', XRX.numpy())

    # calculate pseudo inverse via SVD
    # method 1
    # slightly better than tfp.math.pinv when L2_param=0
    XRX_pinv = pinv_naive(XRX)

    # method 2
    # XRX_pinv = pinv(XRX)

    # w = w - (X^T R X)^(-1) X^T (mu-y)
    # w_new = tf.assign(w_old, w_old - tf.matmul(tf.matmul(XRX_pinv, tf.transpose(X)), mu - y))
    y = tf.cast(y, tf_dtype)
    w_update = tf.matmul(XRX_pinv, tf.matmul(tf.transpose(X), mu - y) + L2_param * w_old)
    return w_update


def optimize(w_old, w_update):
    """custom update op, instead of using SGD variants"""
    return w_old.assign(w_old - w_update)


def train_IRLS(X_train, y_train, X_test=None, y_test=None, L2_param=0, max_iter=MAX_ITER):
    """train Logistic Regression via IRLS algorithm
    X: Nxd
    y: Nx1
    ---
    """
    N, d = X_train.shape
    w = tf.Variable(0.01 * tf.ones((d, 1), dtype=tf.float32), name="w")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_writer = tf.summary.create_file_writer(f"./logs/{current_time}")
    print("start training...")
    print("L2 param(lambda): {}".format(L2_param))
    i = 0
    # iteration
    while i <= max_iter:
        print("iter: {}".format(i))

        # print('\t neg log likelihood: {}'.format(sess.run(neg_L, feed_dict=train_feed_dict)))
        neg_L = neg_log_likelihood(w, X_train, y_train, L2_param)
        print("\t neg log likelihood: {}".format(neg_L))
        train_acc = compute_acc(X_train, y_train, w)
        with summary_writer.as_default():
            tf.summary.scalar("train_acc", train_acc, step=i)
            tf.summary.scalar("train_neg_L", neg_L, step=i)

        test_acc = compute_acc(X_test, y_test, w)
        with summary_writer.as_default():
            tf.summary.scalar("test_acc", test_acc, step=i)
        print("\t train acc: {}, test acc: {}".format(train_acc, test_acc))

        L2_norm_w = np.linalg.norm(w.numpy())
        print("\t L2 norm of w: {}".format(L2_norm_w))

        if i > 0:
            diff_w = np.linalg.norm(w_update.numpy())
            print("\t diff of w_old and w: {}".format(diff_w))
            if diff_w < 1e-2:
                break
        w_update = update(w, X_train, y_train, L2_param)
        w = optimize(w, w_update)
        i += 1
    print("training done.")


if __name__ == "__main__":
    # test_acc should be about 0.85
    lambda_ = 20  # 0
    train_IRLS(X_train, y_train, X_test, y_test, L2_param=lambda_, max_iter=100)

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train.reshape(N_train,))
    y_pred_train = classifier.predict(X_train)
    train_acc = np.sum(y_train.reshape(N_train,) == y_pred_train)/N_train
    print('train_acc: {}'.format(train_acc))
    y_pred_test = classifier.predict(X_test)
    test_acc = np.sum(y_test.reshape(N_test,) == y_pred_test)/N_test
    print('test acc: {}'.format(test_acc))
