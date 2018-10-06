from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_rsp = x.reshape(N, -1)           # get (N, D) 
    out = x_rsp.dot(w) + b             # get (N, M)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)                  # 三个变量不要改变
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout.dot(w.T).reshape(x.shape)        # (N, D) -> (N, d1, ..., d_k)
    
    x_rsp = x.reshape(N, -1)                   # get (N, D)
    dw = np.dot(x_rsp.T, dout)                 # get (D, M)

    db = np.sum(dout, axis=0)                  # get (M,  )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)        # 进行max(0, x)操作
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    mask = x > 0      # 获取大于0的位置，用True表示
    dx = dout*mask    # 按元素乘，将梯度分配给大于0的位置
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))   # 没有该属性取0
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)   # 把每列求平均 get (D, )
        sample_var = np.var(x, axis=0)     # 每列求方差 get   (D, )
        den = np.sqrt(sample_var + eps)
        x_norm = (x - sample_mean) / den   # 归一化得到0均值，1方差的分布 (N, D)
        out = x_norm * gamma + beta        # 进行缩放和平移
        # cache = gamma * den              # 输出对输入求导只保留系数
        cache = (gamma, sample_mean, sample_var, eps, x, x_norm) 

        # 保存均值和方差用于测试阶段
        running_mean = momentum * running_mean + (1-momentum)* sample_mean
        running_var = momentum * running_var + (1-momentum)* sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm_test = (x - running_mean) / np.sqrt(running_var + eps)   # 使用训练阶段的mean和var归一化
        out = gamma * x_norm_test + beta                                # 缩放和平移
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    gamma, sample_mean, sample_var, eps, x, x_norm = cache
    dgamma = np.sum(dout * x_norm, axis=0)    # (D, )   # gamma是常数，所以将样本个数维度叠加
    dbeta  = np.sum(dout, axis=0)             # (D, )
    # 均值和方差中也有x，所以也是x的函数
    m, D = x.shape 

    # 方法一、 使用计算图表示算式后，按不同的节点求导 m与N一致  (部分中间节点)
    # dl_xnorm = dout * gamma
    # dx_mu_1 = dl_xnorm / np.sqrt(sample_var+eps)                     #  (N,D)
    # dl_var = 2/m*(x-sample_mean) * np.sum(-1/2 *dl_xnorm*(x-sample_mean)* (sample_var+eps)**(-3/2), axis=0)  # (D,  )
    # dx_mu_2 = dl_var
    # dx_mu = dx_mu_1 + dx_mu_2                                        # (N, D)
    # dx_1 = -1/m * np.sum(dx_mu, axis=0)                              # (D, )
    # dx_2 = dx_mu
    # dx = dx_1 + dx_2

    # 全部中间节点
    dl_xhat = dout * gamma                                             # (N, D)
    dl_mu1 = dl_xhat / np.sqrt(sample_var+eps)                         # 对分子上均值的偏导 (N, D)
    dl_den = np.sum(-dl_xhat *(x-sample_mean) / (sample_var+eps), axis=0)       # 对分母求偏导 (D, )
    dl_var = 1/2 * dl_den / np.sqrt(sample_var+eps)                    # 对方差求偏导 (D, )
    dl_mu_squ = np.tile(dl_var, (m, 1)) / m                            # 对均值的平方求导 (N, D) 梯度分别分配到m维度
    dl_mu2 = dl_mu_squ * 2*(x-sample_mean)                             # 对分母上均值的偏导 (N, D)
    dl_x_mu = dl_mu1 + dl_mu2                                          # 将x-mu的梯度累加  (N, D)
    dl_mu = -np.sum(dl_x_mu, axis=0)                                   # 对mu的梯度        (D, )
    # dl_x1 = np.tile(dl_mu, (m, 1)) / m                               # 对x的部分梯度      (N, D)
    dl_x1 = np.ones((m, D)) * dl_mu / m                                # 与上一句功能一样
    dl_x2 = dl_x_mu                                                    # 对x的另一部分梯度  (N, D)
    dx = dl_x1 + dl_x2                                                 # 对x的梯度         (N, D)

    # 方法二、按求导公式
    # https://blog.csdn.net/zhangxb35/article/details/69568456
    # dl_xhat = dout * gamma
    # dl_var = np.sum(-1/2 * dl_xhat * (x-sample_mean) * (sample_var+eps)**(-3/2), axis=0)
    # dl_mean = np.sum(-dl_xhat / np.sqrt(sample_var+eps), axis=0) + dl_var * -2/m * np.sum(x-sample_mean, axis=0)
    # dx = dl_xhat/np.sqrt(sample_var+eps) + 2/m * dl_var*(x-sample_mean) + dl_mean / m
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    gamma, sample_mean, sample_var, eps, x, x_norm = cache
    dgamma = np.sum(dout * x_norm, axis=0)    # (D, )   # gamma是常数，所以将样本个数维度叠加
    dbeta  = np.sum(dout, axis=0)             # (D, )
    # 均值和方差中也有x，所以也是x的函数
    m = x.shape[0]

    # 方法二、按求导公式
    # https://blog.csdn.net/zhangxb35/article/details/69568456
    dl_xhat = dout * gamma
    dl_var = np.sum(-1/2 * dl_xhat * (x-sample_mean) * (sample_var+eps)**(-3/2), axis=0)
    dl_mean = np.sum(-dl_xhat / np.sqrt(sample_var+eps), axis=0) + dl_var * -2/m * np.sum(x-sample_mean, axis=0)
    dx = dl_xhat/np.sqrt(sample_var+eps) + 2/m * dl_var*(x-sample_mean) + dl_mean / m
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    x = x.T                            # (D, N) 将x转置复用BN的程序
    sample_mean = np.mean(x, axis=0)   # 把每列求平均 get (N, )
    sample_var = np.var(x, axis=0)     # 每列求方差 get   (N, )
    den = np.sqrt(sample_var + eps)
    x_norm = (x - sample_mean) / den                            # 归一化得到0均值，1方差的分布 (D, N)
    out = x_norm * gamma.reshape(-1,1) + beta.reshape(-1,1)     # 进行缩放和平移
    out = out.T                                                 # 将输出的维度与输入的数据对应 (N, D)
    # cache = gamma * den                                       # 输出对输入求导只保留系数
    x = x.T                                                     # 将数据维度还原 (N, D)
    x_norm = x_norm.T
    cache = (gamma, sample_mean, sample_var, eps, x, x_norm) 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################

    gamma, sample_mean, sample_var, eps, x, x_norm = cache
    gamma = gamma.reshape(-1,1)
    dgamma = np.sum(dout * x_norm, axis=0)    # (D, )   # gamma是常数，所以将样本个数维度叠加
    dbeta  = np.sum(dout, axis=0)             # (D, )

    dout = dout.T                                       # !!!转置一下复用上述的BN程序!!!
    x = x.T                                             # (D, N)
    x_norm = x_norm.T    
    # 均值和方差中也有x，所以也是x的函数
    m, D = x.shape                                      # 这里的m表示维度，D表示样本个数

    dl_xhat = dout * gamma
    dl_var = np.sum(-1/2 * dl_xhat * (x-sample_mean) * (sample_var+eps)**(-3/2), axis=0)
    dl_mean = np.sum(-dl_xhat / np.sqrt(sample_var+eps), axis=0) + dl_var * -2/m * np.sum(x-sample_mean, axis=0)
    dx = dl_xhat/np.sqrt(sample_var+eps) + 2/m * dl_var*(x-sample_mean) + dl_mean / m    # (D, N)
    dx = dx.T                                           # 转置还原维度(N, D)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not        
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # 参数的*号代表把元组的元素取出来 (a,b) => a,b
        mask = (np.random.rand(*x.shape) < p) / p    # p是输出激活的概率
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x                                     # 输出保持不变
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape                                       # 将数据的维度表示为变量
    F, C, HH, WW = w.shape
    stride = conv_param.get('stride', 1)                       # 获取步长，如果不存在则返回1
    pad = conv_param.get('pad', 0)                             # 获取padding，如果不存在则返回0
    # 如果需要进行padding操作,只对(N, C, H, W)的后两维度进行padding操作
    if pad != 0:                                                         
        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    H_new = int((H + 2 * pad - HH) / stride + 1)                              # 计算卷积后的尺寸大小
    W_new = int((W + 2 * pad - WW) / stride + 1)

    # 卷积过程              
    out = np.zeros((N, F, H_new, W_new))                                      # 创建输出维度的零矩阵
    for num in range(N):
        for filter in range(F):                                               # 卷积核数
            for height in range(H_new):                                       # 新feature map的高
                for width in range(W_new):                                    # 新feature map的宽
                    start_h = stride * height
                    start_w = stride * width 
                    neural_prod = np.sum(x_pad[num, :, start_h:start_h+HH, start_w:start_w+WW] * w[filter, :, :, :])
                    out[num,filter,height, width] = neural_prod + b[filter]   # 切记加偏置
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    (x, w, b, conv_param) = cache                              
    dx = np.zeros(x.shape)                                     # (N,C,H,W)
    dw = np.zeros(w.shape)                                     # (F,C,H,W)
    db = np.zeros(b.shape)                                     # (F, )
    N, C, H, W = x.shape                                       # 将数据的维度表示为变量
    F, C, HH, WW = w.shape
    stride = conv_param.get('stride', 1)                       # 获取步长，如果不存在则返回1
    pad = conv_param.get('pad', 0)                             # 获取padding，如果不存在则返回0
    # 如果需要进行padding操作,只对(N, C, H, W)的后两维度进行padding操作
    if pad != 0:                                                         
        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    dx_pad = np.zeros(x_pad.shape)                            # (N,C,H',W')
    H_new = int((H + 2 * pad- HH) / stride + 1)               # 计算卷积后的尺寸大小
    W_new = int((W + 2 * pad- WW) / stride + 1)   

    # print('H W:', H, W)
    # print('H_new W_new', H_new, W_new)
    # dout (N,F,H,W) 维度
    # mask = np.ones(*w)                                         # 定义一个与滤波器同维度的矩阵
    for num in range(N):
        for filter in range(F):                                               # 卷积核数
            for height in range(H_new):                                       # 新feature map的高
                for width in range(W_new):                                    # 新feature map的宽
                    start_h = stride * height
                    start_w = stride * width
                    # (1,C,HH,WW)
                    # print('1',w[filter,:,:,:].shape)
                    # # print(dout[num,filter,height,width])
                    # print('2',(w[filter,:,:,:] * dout[num,filter,height,width]).shape)
                    # print('3',dx[num, :, start_h:start_h+HH, start_w:start_w+WW].shape)
                    # print('---------------')
                    dx_pad[num, :, start_h:start_h+HH, start_w:start_w+WW] += w[filter,:,:,:] * dout[num,filter,height,width]
                    dw[filter,:,:,:] +=  x_pad[num, :, start_h:start_h+HH, start_w:start_w+WW] * dout[num,filter,height,width]
                    db[filter] += dout[num,filter,height,width]
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    # print(dx.shape)
    # print(dw.shape)
    # print(db.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    H_new = int((H - pool_height) / stride + 1)             # 计算输出的尺寸
    W_new = int((W - pool_width) / stride + 1)

    out = np.zeros((N, C, H_new, W_new))                    # 创建输出为0的矩阵
    for height in range(H_new):
        for width in range(W_new):
            start_h = stride * height                       # 计算pool的起始位置
            start_w = stride * width
            x_masked = x[:,:,start_h:start_h+pool_height,start_w:start_w+pool_width]   # (N,C,H',W')
            # print('1',x_masked.shape)
            # print('2',out.shape)
            # print('----')
            out[:, :, height, width] = np.max(x_masked, axis=(2,3))                    # 在H与W维度中找最大值
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    H_new = int((H - pool_height) / stride + 1)                    # 计算输出的尺寸
    W_new = int((W - pool_width) / stride + 1)

    #----------------------------------方法一---------------------------------------
    # dout (N,C,H',W')
    # dx = np.zeros_like(x)                                        # 创建与x同维度的零矩阵
    # for num in range(N):
    #     for channel in range(C):
    #         for height in range(H_new):
    #             for width in range(W_new):
    #                 start_h = stride * height                    # 计算pool的起始位置
    #                 start_w = stride * width
    #                 x_masked = x[num,channel,start_h:start_h+pool_height,start_w:start_w+pool_width]   # (1,1,H',W') 
    #                 # print('1',x_masked.shape)
    #                 # print('--------')
    #                 # ------------------方法一---------------------
    #                 # 使用max与where找出最大值索引
    #                 max_value = np.max(x_masked)                # 在H与W维度中找最大值
    #                 max_index = np.where(x==max_value)          # 找到最大值的索引
    #                 # print(max_index)
    #                 dx[max_index] += dout[num,channel,height,width]    # (N,C)

    #                 # ------------------方法二---------------------
    #                 # 使用argmax找到索引
    #                 # 获取当前x区块的索引，需要叠加start_h/start_w获取x的最终索引
    #                 # local_ind_h, local_ind_w = np.unravel_index(np.argmax(x_masked, axis=None), x_masked.shape)  
    #                 # dx[num,channel,start_h+local_ind_h,start_w+local_ind_w] +=  dout[num,channel,height,width]            
    
    #---------------------------------方法二(更佳)------------------------------------
    dx = np.zeros_like(x)
    for height in range(H_new):
        for width in range(W_new):
            start_h = stride * height                                                  # 计算pool的起始位置
            start_w = stride * width
            x_masked = x[:,:,start_h:start_h+pool_height,start_w:start_w+pool_width]   # (N,C,H',W')
            max_x_masked = np.max(x_masked, axis=(2,3))                                # 在H和W维度上取最大值  (N,C)
            # 使用None来扩展维度，时 (N,C) 和 (N,C,H',W')数据可以做运算
            max_index_mask = (x_masked == max_x_masked[:,:,None,None])                 # (N,C,H',W')
            # print(max_index_mask.shape)
            dx[:,:,start_h:start_h+pool_height,start_w:start_w+pool_width] += max_index_mask * dout[:,:,height,width][:,:,None,None]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    # print(N, C, H, W)
    out = np.zeros_like(x)                                      # 创建与x同维度的零矩阵
    x_reshape = x.transpose(0,2,3,1).reshape(-1, C)             # (N, C, H, W) => (N,H,W,C)为了reshape使NHW在一块
    vanilla_bn_out, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param)
    out= vanilla_bn_out.reshape(N, H, W, C).transpose(0,3,1,2)  # (N,H,W,C) => (N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_reshape = dout.transpose(0,2,3,1).reshape(-1, C)  # (N, C, H, W) => (N,H,W,C)为了reshape使NHW在一块    
    dx_reshape, dgamma, dbeta = batchnorm_backward_alt(dout_reshape, cache)
    dx = dx_reshape.reshape(N, H, W, C).transpose(0,3,1,2) # (-1,C) => (N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    N, C, H, W = x.shape
    group_len = C // G
    x_reshape = x.transpose(0,2,3,1).reshape(-1, C)  # (N, C, H, W) => (N,H,W,C)为了reshape使NHW在一块
    out = np.zeros_like(x)                           # (N, C,H,W)
    cache = {'G':G, 'layernorm':[]}
    for slice_g in range(G):
        group_s = slice_g*group_len
        x_grouped = x_reshape[:, group_s:group_s+group_len]        # (N*H*W, group)
        # print(x_grouped.shape)
        # print(gamma[0,0:3,0,0].shape)
        # print(x_grouped.shape, gamma[group_s:group_s+group_len].shape, beta[group_s:group_s+group_len].shape)
        ln_out, temp_cache = layernorm_forward(x_grouped, gamma[0,group_s:group_s+group_len,0,0], beta[0,group_s:group_s+group_len,0,0], gn_param)
        # print(ln_out.shape)
        out[:,group_s:group_s+group_len,:,:] = ln_out.reshape(N,H,W,group_len).transpose(0,3,1,2)
        cache['layernorm'].append(temp_cache)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    N, C, H, W = dout.shape
    dx = np.zeros_like(dout)                                         # 创建与dout同维度的零矩阵
    dgamma = np.zeros((1,C,1,1))
    dbeta = np.zeros((1,C,1,1))
    G, ln_cache_list = cache['G'], cache['layernorm']
    group_len = C // G
    dout_reshape = dout.transpose(0,2,3,1).reshape(-1, C)            # 坐标轴变为(N,H,W,C)方便reshape

    for slice_g in range(G):
        group_s = slice_g*group_len
        # 每次处理C维度中的部分维度，加H与W的全部维度
        dout_grouped = dout_reshape[:, group_s:group_s+group_len]        # (N*H*W, group)
        dx_reshape, dgamma[0,group_s:group_s+group_len,0,0], dbeta[0,group_s:group_s+group_len,0,0] = layernorm_backward(dout_grouped, ln_cache_list[slice_g])
        # print(ln_out.shape)
        dx[:,group_s:group_s+group_len,:,:] = dx_reshape.reshape(N,H,W,group_len).transpose(0,3,1,2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)  # max(0, S_j - S_y_i + 1)
    margins[np.arange(N), y] = 0                                            # 将正确位置置零
    loss = np.sum(margins) / N

    num_pos = np.sum(margins > 0, axis=1)            # 每行内(每个样本)激活位置的个数
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos                   # 使y_i被计算的次数与S_j一样多
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    shifted_logits = x - np.max(x, axis=1, keepdims=True)       # 减去每行内的最大值，防止出现数值稳定问题
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)   # 每行的数据取exp并求和
    log_probs = shifted_logits - np.log(Z) 
    loss = -np.sum(log_probs[np.arange(N), y]) / N              # -f_y_i + log(sum_e^(f_j))

    probs = np.exp(log_probs)                                   # 取exp将对数化开到分母，分子为e_j   
    dx = probs.copy()
    dx[np.arange(N), y] -= 1                                    # 减去正确类多的-1
    dx /= N
    return loss, dx
