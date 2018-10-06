from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        C, H, W = input_dim
        # 卷积层的权重初始化 (F,C,HH,WW)
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        # 全连接层的权重初始化，卷积与池化参数使输入的数据尺寸减半 (hidden_dim, num_filters*H*W/4)
        # 注意权重的维度 (L-1, L)
        hidden1 = num_filters * H * W // 4
        self.params['W2'] = weight_scale * np.random.randn(hidden1, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # 输出层权重初始化 (num_classes, hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        cache = {}        # 用来保存前向传播得到的值用于反向传播
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # 卷积层
        conv1_out, conv1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # 全连接层
        # affine_forward函数中已经将该数据维度变为 (N, -1) ,所以该句可以去掉
        # conv1_out_flatten = conv1_out.reshape(conv1_out.shape[0], -1)  # 将卷积的feature map拉成一维向量
        fc2_out, fc2_cache = affine_relu_forward(conv1_out, W2, b2)
        # 输出层
        fc3_out, fc3_cache =  affine_forward(fc2_out, W3, b3)
        scores = fc3_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 损失函数：数据损失+正则化损失
        loss, softmax_grads= softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

        # 反向传播计算梯度
        # 第3层输出层
        upstream_grads, grads['W3'], grads['b3'] = affine_backward(softmax_grads, fc3_cache)
        # 第2层全连接层
        upstream_grads, grads['W2'], grads['b2'] = affine_relu_backward(upstream_grads, fc2_cache)
        # 第1层卷积层 (N,F,H',W')，梯度需要reshape
        # print(conv1_out.shape)
        N, F, H1, W1 = conv1_out.shape                          # 获取卷积输出的尺寸
        # upstream_grads = upstream_grads.reshape(N, F, H1, W1)   # 将全连接层的梯度reshape为卷积可处理维度
        upstream_grads, grads['W1'], grads['b1'] = conv_relu_pool_backward(upstream_grads, conv1_cache)
       
        for num in range(1,4):                                  # 切记梯度有正则化项
            grads['W%d'%num] += self.reg * self.params['W%d'%num]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads
