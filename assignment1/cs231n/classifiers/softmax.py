import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)                                        # NxC dimendional
  scores = scores - np.max(scores, axis=1).reshape(-1,1)   # 每一行减去自己行的最大值
  # scores = scores - np.max(scores)
  for i in range(num_train):
  	add_den = np.sum(np.exp(scores[i,:]))                  # 第i行分数求分母 
  	loss_i = -scores[i, y[i]] + np.log(add_den)            # 注意scores[i,y[i]]
  	loss += loss_i
  	# print(X[0,:].reshape(-1,1).shape, dW[:,0].reshape(-1,1).shape)
  	for j in range(num_classes):
  		dW[:,j:j+1] += X[i,:].reshape(-1,1)*np.exp(scores[i,j]) / add_den  # 把不正确类的导数累加，每个样本C次
  	dW[:, y[i]:y[i]+1] += -X[i,:].reshape(-1,1)                           # 把正确类的导数累加，每个样本累加1次
  
 
  loss = loss / num_train + 0.5*reg*np.sum(W*W)
  dW = dW / num_train + reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  W: DxC
  X: NxD
  y: (N,)
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)                                                     # NxC
  scores = scores - np.max(scores, axis=1).reshape(-1,1)                # 为了数值稳定性，每行减去最大值
  loss += np.sum(np.log(np.sum(np.exp(scores), axis=1)))                # 累加非正确类的损失
  loss += -np.sum(scores[range(num_train), y])                          # 累加正确类的损失
  loss = loss / num_train + 0.5*reg*np.sum(W*W)                         # 取平均并累加正则项


  false_label= np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1,1)    # NxC 非正确类的梯度
  dW += np.dot(X.T, false_label)                                 # Dxc
  margin = np.zeros((num_train, num_classes))                   
  margin[range(num_train), y] = -1
  dW += np.dot(X.T, margin)                                      # DxC  正确类的梯度
  dW = dW / num_train + reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

