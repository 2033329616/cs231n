import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero  (D, C)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):              # N个样本
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):          # C个类别
      if j == y[i]:       
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:          # 有损失的地方就有梯度
        loss += margin
        dW[:, j:j+1] += X[i].reshape(-1, 1)             # 对w_j偏导部分
        dW[:,y[i]:y[i]+1] += -X[i].reshape(-1, 1)       # 对w_y_i偏导部分



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)        # 直接*表示矩阵相应元素的乘积
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  W:(D, C)
  X:(N, D)
  y:(N,)  y[i]表示对于X[i]的标签
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # ---------------------------------方法一 ---------------------------------- 
  scores = X.dot(W)                                # get NxC
  # 包含了正确分类yi的运算，该位置对应的元素为1
  sub_result = scores - scores[range(y.shape[0]), y].reshape(-1, 1) + 1.0   # get NxC

  positive_num = np.maximum(0, sub_result)
  positive_num[range(X.shape[0]), y] = 0
  loss = np.sum(positive_num) / X.shape[0]  + 0.5*reg*np.sum(W*W)

  # positive_num = sub_result[sub_result > 0]        # 取出大于0的数
  # 多计算了y_i N次，每个样本计算一次，所以总的数减N就行，但由于需要平均，所以总体减1
  # loss = np.sum(positive_num) / X.shape[0] - 1 + 0.5*reg*np.sum(W*W)     # 计算总的损失

  # ---------------------------------方法二 ----------------------------------
  # sub_result[range(y.shape[0]), y] = 0                                  # 将第i行第y_i列的元素置零
  # positive_num = sub_result[sub_result > 0]                             # 将大于0的元素提出来
  # loss = np.sum(positive_num) / X.shape[0] + 0.5*reg*np.sum(W**2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_train = X.shape[0]
  margin = np.zeros(sub_result.shape)             
  margin[sub_result > 0] = 1            # NxC 元素大于0表示需要梯度的地方，也包括y_i

  margin[range(num_train), y] = 0
  margin[range(num_train), y] = -np.sum(margin, axis=1)   # ？？？？？？？？？？

  dW = np.dot(X.T, margin)             # DxC            
  dW = dW / num_train + reg*W

  # margin[margin>0]=1
  # margin[margin<=0]=0

  # row_sum = np.sum(margin, axis=1)                  # 1 by N
  # margin[np.arange(X.shape[0]), y] = -row_sum
  # dW += np.dot(X.T, margin)     # D by C

  # dW/=X.shape[0]
  # dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
