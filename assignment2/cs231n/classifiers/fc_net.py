from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)   # (D, H)
        self.params['b1'] = np.zeros(hidden_dim)                                    # (H, )
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes) # (H, C)
        self.params['b2'] = np.zeros(num_classes)                                   # (C, )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # affine - relu - affine - softmax
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        h1_out, h1_cache = affine_relu_forward(X, W1, b1)   # 获取第一层的输出：affine + relu
        o2_out, o2_cache = affine_forward(h1_out, W2, b2)   # 获取第二层的输出：affine
        scores = o2_out                                     # (N, C)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, grads_softmax = softmax_loss(scores, y)                    # 获取softmax的输出,数据损失和梯度
        regular_loss = 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))  # 获取正则化损失
        loss += regular_loss                                             # 总的损失

        # 梯度反向传播，本层的梯度计算需要用到本层前向传播的值
        do2, dW2, db2 = affine_backward(grads_softmax, o2_cache) # 第二层梯度反向传播
        dh1, dW1, db1 = affine_relu_backward(do2, h1_cache)      # 第一层梯度反向传播
        dW1 += self.reg * W1                                     # 添加正则化的损失
        dW2 += self.reg * W2
        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        for num in range(self.num_layers):
            if num == 0:                            # 第一层的权重
                rows = input_dim                    # (D, H)
                cols = hidden_dims[num]             
            elif num == (self.num_layers -1):  # 最后一层的权重
                rows = hidden_dims[num-1]           # (H, C)
                cols = num_classes                  
            else:                                   # 中间的权重
                rows = hidden_dims[num-1]
                cols = hidden_dims[num]             # (H1, H2)

            self.params['W' + str(num+1)] = weight_scale * np.random.randn(rows, cols)   # (H1, H2)
            self.params['b' + str(num+1)] = np.zeros(cols)                               # (H2)  

            # 初始化bn层
            if (self.normalization == 'batchnorm') and (num != self.num_layers-1):     # 最后一层无BN
                self.params['gamma%d'%(num+1)] = np.ones(hidden_dims[num])             # 与隐层单元数一致 
                self.params['beta%d'%(num+1)] = np.zeros(hidden_dims[num]) 
            if (self.normalization == 'layernorm') and (num != self.num_layers-1):     # 最后一层无LN
                self.params['gamma%d'%(num+1)] = np.ones(hidden_dims[num])             # 与隐层单元数一致 
                self.params['beta%d'%(num+1)] = np.zeros(hidden_dims[num])      
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':                # ????为什么不同的批次传递???
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:             # 怎么处理训练和测试过程?????
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.  ???????                                                     #
        ############################################################################
        cache = {}   # 来保存前向传播的值，用于之后的反向传播
        L = self.num_layers
        layer_input = X
        for num in range(L-1):                                     # L-1层  
            W = self.params['W%d'% (num+1)]                        # 使用权重的名称来访问参数
            b = self.params['b%d'% (num+1)]
            # h_out = 'h%d_out'% (num+1)                           # 为输出定义动态变量名称
            h_cache = 'h%d_cache'% (num+1)
            # relu_out = 'relu%d_out'% (num+1)
            relu_cache = 'relu%d_cache'% (num+1)

            h_out, cache[h_cache] = affine_forward(layer_input, W, b)      # 全连接层前向传播
            if self.normalization=='batchnorm':
                bn_cache = 'bn%d_cache'%(num+1)
                bn_out, cache[bn_cache] = batchnorm_forward(h_out, self.params['gamma%d'%(num+1)], 
                                                            self.params['beta%d'%(num+1)], self.bn_params[num])
                h_out = bn_out
            if self.normalization=='layernorm':
                ln_cache = 'ln%d_cache'%(num+1)
                ln_out, cache[ln_cache] = layernorm_forward(h_out, self.params['gamma%d'%(num+1)], 
                                                            self.params['beta%d'%(num+1)], self.bn_params[num])
                h_out = ln_out

            relu_out, cache[relu_cache] = relu_forward(h_out)              # 激活函数处理
            layer_input = relu_out                                         # 更新下次传入的数据
        # 计算第L层的输出，无激活函数
        h_out, cache['h%d_cache'% (L)] = affine_forward(layer_input, self.params['W%d'% (L)], self.params['b%d'% (L)])
        scores = h_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, softmax_grads = softmax_loss(scores, y)               # 获取softmax的输出,数据损失和梯度
        # 先将第L层反向传播，因为该层无激活函数
        upstream_grads, grads['W%d'%L], grads['b%d'%L] = affine_backward(softmax_grads, cache['h%d_cache'% (L)])
        grads['W%d'%L] += self.reg * self.params['W%d'%L]                # 累加正则化部分
        regular_loss = np.sum(self.params['W%d'%L]*self.params['W%d'%L]) # 正则化损失

        for num in list(range(L-1))[::-1]:                          # 反向传播前L-1层，倒序访问网络
            W = 'W%d'% (num+1)
            b = 'b%d'% (num+1)
            regular_loss += np.sum(self.params[W]*self.params[W])   # 累积正则化损失
            # 反向传播ReLU层
            upstream_grads = relu_backward(upstream_grads, cache['relu%d_cache'% (num+1)])

            # 反向传播BN层
            if self.normalization=='batchnorm':
                gamma = 'gamma%d'%(num+1)
                beta = 'beta%d'%(num+1)
                upstream_grads, grads[gamma], grads[beta]= batchnorm_backward_alt(upstream_grads, cache['bn%d_cache'%(num+1)])
            # 反向传播LN层
            if self.normalization=='layernorm':
                gamma = 'gamma%d'%(num+1)
                beta = 'beta%d'%(num+1)
                upstream_grads, grads[gamma], grads[beta]= layernorm_backward(upstream_grads, cache['ln%d_cache'%(num+1)])

            # 反向传播全连接层
            upstream_grads, grads[W], grads[b] = affine_backward(upstream_grads, cache['h%d_cache'% (num+1)])
            grads[W] += self.reg * self.params[W]  # 累加正则化梯度部分

        loss +=  0.5 * self.reg * regular_loss     # 总的损失函数

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
