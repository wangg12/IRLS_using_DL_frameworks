# python 3
from __future__ import print_function, division, absolute_import

import os
import argparse
import random
import numpy as np
# from numpy import linalg

from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
#from scipy.sparse import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

path_train = '../a9a/a9a'
path_test = '../a9a/a9a.t'
MAX_ITER = 100
np_dtype = np.float32
tf_dtype = tf.float32

# manual seed
manualSeed = random.randint(1, 10000) # fix seed
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
X_train = np.hstack((np.ones((N_train, 1)), X_train.toarray()))
X_test = np.hstack((np.ones((N_test, 1)), X_test.toarray()))


y_train = y_train.reshape((N_train,1))
y_test = y_test.reshape((N_test,1))

# label: -1, +1 ==> 0, 1
y_train = np.where(y_train==-1, 0, 1)
y_test = np.where(y_test==-1, 0, 1)

# NB: here X's shape is (N,d), which differs to the derivation


# def sigmoid(v, a=1):
#   '''
#   1./(1+exp(-a*v))
#   v: input, can be a ndarray, in this case, sigmoid is applied element-wise
#   '''
#   res = np.zeros(v.shape, dtype=dtype)
#   res = np.where(a*v>=0, 1./(1+np.exp(-a*v)), np.exp(a*v)/(1 + np.exp(a*v)))
#   return res #1./(1+np.exp(-a*v))


def neg_log_likelihood(w, X, y, L2_param=None):
  '''
  w: dx1
  X: Nxd
  y: Nx1
  L2_param: \lambda>0, will introduce -\lambda/2 ||w||_2^2
  '''
  res = tf.matmul(tf.matmul(tf.transpose(w), tf.transpose(X)), y) - tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(X, w))))
  if L2_param != None and L2_param > 0:
    res += -0.5*L2_param*tf.matmul(tf.transpose(w), w)
  return -res[0][0]


def prob(X,w):
  '''
  X: Nxd
  w: dx1
  ---
  prob: N x num_classes(2)'''
  y = tf.constant(np.array([0., 1.]), dtype=tf.float32)
  prob = tf.exp(tf.matmul(X, w) * y)/(1+tf.exp(tf.matmul(X, w)))
  return prob


def compute_acc(X, y, w):
  p = prob(X, w)
  y_pred = tf.cast(tf.argmax(p, axis=1), tf.float32)
  y = tf.squeeze(y)
  acc = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
  return acc


def update(w_old, X, y, L2_param=0):
  '''
  w_new = w_old - w_update
  w_update = (X'RX+lambda*I)^(-1) (X'(mu-y) + lambda*w_old)
  lambda is L2_param

  w_old: dx1
  X: Nxd
  y: Nx1
  ---
  w_update: dx1
  '''
  d = X.shape.as_list()[1]
  mu = tf.sigmoid(tf.matmul(X, w_old)) # Nx1

  R_flat = mu * (1 - mu) # element-wise, Nx1

  L2_reg_term = L2_param * tf.eye(d)
  XRX = tf.matmul(tf.transpose(X), R_flat*X) + L2_reg_term  # dxd

  S,U,V = tf.svd(XRX, full_matrices=True, compute_uv=True)
  S = tf.expand_dims(S, 1)

  # calculate pseudo inverse via SVD
  S_pinv = tf.where(tf.not_equal(S, 0),
                    1/S,
                    tf.zeros_like(S)) # not good, will produce inf when divide by 0
  XRX_pinv = tf.matmul(V, S_pinv*tf.transpose(U))

  # w = w - (X^T R X)^(-1) X^T (mu-y)
  #w_new = tf.assign(w_old, w_old - tf.matmul(tf.matmul(XRX_pinv, tf.transpose(X)), mu - y))
  w_update = tf.matmul(XRX_pinv, tf.matmul(tf.transpose(X), mu - y) + L2_param*w_old)
  return w_update


def optimize(w_old, w_update):
  '''custom update op, instead of using SGD variants'''
  return w_old.assign(w_old - w_update)


def train_IRLS(X_train, y_train, X_test=None, y_test=None, L2_param=0, max_iter=MAX_ITER):
  '''train Logistic Regression via IRLS algorithm
  X: Nxd
  y: Nx1
  ---

  '''
  N,d = X_train.shape
  X = tf.placeholder(dtype=tf.float32, shape=(None, 124), name="X")
  y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

  w = tf.Variable(0.01*tf.ones((d,1), dtype=tf.float32), name='w')
  w_update = update(w, X, y, L2_param)
  with tf.variable_scope('neg_L'):
    neg_L = neg_log_likelihood(w, X, y, L2_param)
  neg_L_summ = tf.summary.scalar('neg_L', neg_L)

  with tf.variable_scope('accuracy'):
    acc = compute_acc(X, y, w)
  acc_summ = tf.summary.scalar('acc', acc)

  optimize_op = optimize(w, w_update)

  merged_all = tf.summary.merge_all()

  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) 
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.8

  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter('./log', sess.graph)

  train_feed_dict = {X:X_train, y:y_train}
  test_feed_dict = {X:X_test, y:y_test}

  print('start training...')
  print('L2 param(lambda): {}'.format(L2_param))
  i = 0
  # iteration
  w_old = w
  while i <= max_iter:
    print('iter: {}'.format(i))

    print('\t neg log likelihood: {}'.format(sess.run(neg_L, feed_dict=train_feed_dict)))

    train_acc, merged = sess.run([acc, merged_all], feed_dict=train_feed_dict)
    summary_writer.add_summary(merged, i)

    test_acc  = sess.run(acc, feed_dict=test_feed_dict)
    print('\t train acc: {}, test acc: {}'.format(train_acc, test_acc))

    L2_norm_w = np.linalg.norm(sess.run(w))
    print('\t L2 norm of w: {}'.format(L2_norm_w))

    if i>0:
      diff_w = np.linalg.norm(sess.run(w_update, feed_dict=train_feed_dict))
      print('\t diff of w_old and w: {}'.format(diff_w))
      if diff_w < 1e-2:
        break

    w_new = sess.run(optimize_op, feed_dict=train_feed_dict)
    i += 1
  print('training done.')


if __name__=='__main__':
  lambda_ = 20 # 0
  train_IRLS(X_train,y_train,X_test,y_test,L2_param=lambda_,max_iter=100)

  # from sklearn.linear_model import LogisticRegression
  # classifier = LogisticRegression()
  # classifier.fit(X_train, y_train.reshape(N_train,))
  # y_pred_train = classifier.predict(X_train)
  # train_acc = np.sum(y_train.reshape(N_train,) == y_pred_train)/N_train
  # print('train_acc: {}'.format(train_acc))
  # y_pred_test = classifier.predict(X_test)
  # test_acc = np.sum(y_test.reshape(N_test,) == y_pred_test)/N_test
  # print('test acc: {}'.format(test_acc))
