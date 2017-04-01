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

from torch.autograd import Variable

from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
#from scipy.sparse import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


path_train = '../a9a/a9a'
path_test = '../a9a/a9a.t'
MAX_ITER = 100
np_dtype = np.float32
use_cuda = False
if torch.cuda.is_available():
  use_cuda = True


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

# X_train = csr_matrix(X_train, dtype=dtype)
# X_test = csr_matrix(X_test, dtype=dtype)

y_train = y_train.reshape((N_train,1))
y_test = y_test.reshape((N_test,1))

# label: -1, +1 ==> 0, 1
y_train = np.float32(np.where(y_train==-1, 0, 1))
y_test = np.float32(np.where(y_test==-1, 0, 1))


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
  res = torch.mm(torch.mm(w.data.t(), X.t()), y) - torch.sum(torch.log(1 + torch.exp(torch.mm(X, w.data))))
  if L2_param != None and L2_param > 0:
    res += -0.5*L2_param*torch.mm(w.data.t(), w.data)
  return -res


def prob(X,w):
  '''
  X: Nxd
  w: dx1
  ---
  prob: N x num_classes(2)'''

  y = torch.FloatTensor(np.array([[0., 1.]])) # 1x2
  if use_cuda:
    y = y.cuda()
  Xw = X.mm(w.data) # Nx1

  prob = torch.exp(Xw.expand(Xw.size()[0], 2)
                    * y.expand(Xw.size()[0], 2)) / (1+torch.exp(X.mm(w.data))).expand(Xw.size()[0],2)
  return prob


def compute_acc(X, y, w):
  p = prob(X, w)
  _, y_pred = torch.max(p, 1)
  y_pred = y_pred.type_as(torch.FloatTensor())
  if use_cuda:
    y_pred = y_pred.cuda()
  acc = torch.mean( (y==y_pred).type_as( y_pred))
  return acc


def update_weight(w_old, X, y, L2_param=0):
  '''
  w_new = w_old - w_update
  w_update = (X'RX+lambda*I)^(-1) (X'(mu-y) + lambda*w_old)
  lambda is L2_param

  w_old: dx1
  X: Nxd
  y: Nx1
  ---
  w_new: dx1
  '''
  d = X.size()[1]
  mu = torch.sigmoid(X.mm(w_old.data)) # Nx1

  R_flat = mu * (1 - mu) # element-wise, Nx1

  if L2_param == 0:
    L2_param = 1e-2 # don't know why need this to work, this is less numerically stable than tensorflow
  L2_reg_term = L2_param * torch.eye(d)
  if use_cuda:
    L2_reg_term = L2_reg_term.cuda()
  XRX = torch.mm(X.t(), R_flat.expand_as(X)*X) + L2_reg_term  # dxd

  U,S,V = torch.svd(XRX)  
  S = S.unsqueeze(1) # dx1

  # calculate pseudo inverse via SVD
  S_pinv = torch.zeros(S.size()) # init
  if use_cuda:
    S_pinv = S_pinv.cuda()

  S_pinv[S!=0] = 1./S[S!=0]
  XRX_pinv = V.mm(S_pinv.expand_as(U.t()) * U.t())

  # w = w - (X^T R X)^(-1) X^T (mu-y)
  w_update = torch.mm(XRX_pinv, torch.mm(X.t(), mu - y) + L2_param*w_old.data)
  w_new = w_old
  w_new.data = w_old.data - w_update
  return w_new




def train_IRLS(X_train, y_train, X_test=None, y_test=None, L2_param=0, max_iter=MAX_ITER):
  '''train Logistic Regression via IRLS algorithm
  X: Nxd
  y: Nx1
  ---

  '''
  N,d = X_train.shape
  X_train = torch.FloatTensor(X_train)
  X_test = torch.FloatTensor(X_test)
  y_train = torch.FloatTensor(y_train)
  y_test = torch.FloatTensor(y_test)

  if use_cuda:
    X_train = X_train.cuda()
    X_test = X_test.cuda()
    y_train = y_train.cuda()
    y_test = y_test.cuda()

  w = Variable(0.01*torch.ones((d,1)), requires_grad=False)
  if use_cuda:
    w = Variable(0.01*torch.ones((d,1)).cuda(), requires_grad=False)

  print('start training...')
  print('L2 param(lambda): {}'.format(L2_param))
  i = 0
  # iteration
  w_old = w
  while i <= max_iter:
    print('iter: {}'.format(i))

    neg_L = neg_log_likelihood(w_old, X_train, y_train, L2_param)
    print('\t neg log likelihood: {}'.format(neg_L.sum()))

    train_acc = compute_acc(X_train, y_train, w_old)
    test_acc  = compute_acc(X_test, y_test, w_old)
    print('\t train acc: {}, test acc: {}'.format(train_acc, test_acc))

    L2_norm_w = torch.norm(w_old).data.sum()
    print('\t L2 norm of w: {}'.format(L2_norm_w))

    if i>0:
      diff_w = torch.norm(w.data - w_old_data)
      print('\t diff of w_old and w: {}'.format(diff_w))
      if diff_w < 1e-2:
        break

    w_old_data = w.data
    w = update_weight(w_old, X_train, y_train, L2_param)
    # w = update_weight(w, X, y)
    i += 1
  print('training done.')


if __name__=='__main__':
  lambda_ = 0 # 0
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
