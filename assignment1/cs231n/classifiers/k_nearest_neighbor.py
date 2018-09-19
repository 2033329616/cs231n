import numpy as np
from collections import Counter 

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i,j] = np.sqrt(np.sum(np.square(X[i,:] - self.X_train[j,:])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      # Nx3072 -> Nx1 -> 1xN
      # ---------------------------- 方法一 ----------------------------------
      # 使用向量运算，再使用平方与求和运算
      # 可以不使用X[i:i+1,:]把数据变为(1,3072)，直接使用X[i] (3072,)就行
      # dists[i:i+1,:] = np.sqrt(np.sum(np.square(X[i:i+1,:] - self.X_train), axis=1).transpose()) 
      dists[i] = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))
      # ---------------------------- 方法二 ----------------------------------
      # 直接使用向量运算，但有计算浪费，不推荐使用
      # X_reduce = X[i:i+1,:] - self.X_train                      # get Nx3072
      # dists[i] = X_reduce.dot(X_reduce.transpose()).diagonal()  # (N, N)->(N,) 但有运算浪费，因为只取对角线元素
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # split (p-q)^2 to p^2 + q^2 - 2pq
    sum_test = np.sum(X**2, axis=1, keepdims=True)                # get (Nte, 1)
    # print(sum_test.shape)
    sum_train = np.sum(self.X_train**2, axis=1).reshape(1, -1)    # get (1, Ntr)
    # print(sum_train.shape)
    sum_tetr = 2 * X.dot(self.X_train.transpose())                # get (Nte, Ntr)
    # print(sum_tetr.shape)
    dists = np.sqrt(sum_test + sum_train - sum_tetr)              # get (Nte, Ntr)

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      sorted_index = np.argsort(dists[i])[:k]             # 从小到大排列并取前k个
      closest_y = list(self.y_train[sorted_index])       # 获取训练集对应的类别号
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      # ---------------------------- 方法一 -------------------------------
      # 使用Counter统计各个类的投票结果，得到的结果为字典，
      count = Counter(closest_y)              # 计算各类别出现的频率  例如得：{1: 4, 4: 3, 2: 1, ...}
      y_pred[i] = count.most_common(1)[0][0]  # most_common得[(4, 8)]，所以加[0][0]
      # ---------------------------- 方法二 -------------------------------
      # 使用numpy bincount统计各个数据出现次数，使用argmax得到最大值
      count2 = np.bincount(closest_y)         # 结果按[0,num_classes-1]排序
      y_pred[i] = np.argmax(count2)           # 返回结果最大值对应的序号
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

