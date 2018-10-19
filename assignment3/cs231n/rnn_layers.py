from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    h_net = prev_h.dot(Wh) + x.dot(Wx) + b                   # 未激活的隐藏状态 get (N,H)
    # tanh = lambda x:((1-np.exp(-2*x))/(1+np.exp(-2*x)))    # 定义tanh激活函数
    # relu = lambda x: (1 / (1 + np.exp(-x)))
    # # relu = lambda x: (np.exp(x)/(1+np.exp(x)))           # 该方法会出现nan !!!
    # relu_out = relu(2*h_net)                               # (N,H)
    # tanh_out = 2*relu_out - 1                              # tanh激活函数输出 (N,H)

    next_h = np.tanh(h_net)                                  # 进行tanh激活
    cache = (x, prev_h, Wh, Wx, next_h)                    # 为反向传播存储变量
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, Wh, Wx, next_h = cache 
    # relu = lambda x: (1 / (1 + np.exp(-x)))         # 定义ReLU函数计算tanh梯度
    # dtanh = 4*relu_out*(1-relu_out)                 # tanh激活函数的梯度 (N,H)
    dh_net = dnext_h * (1- next_h*next_h)             # 对激活前h_net的梯度   (N,H)

    db = np.sum(dh_net, axis=0)                       # 将梯度按照样本数目累加 (H, )
    dx = dh_net.dot(Wx.T)                             # (N, D)
    dprev_h = dh_net.dot(Wh.T)                        # (N, H)    注意加转置!!!     
    dWx = np.dot(x.T, dh_net)                         # (D, H)
    dWh = np.dot(prev_h.T, dh_net)                    # (H, H)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape                    # (N,T,D) 序列个数，长度，每个向量维度
    H = h0.shape[1]
    h = np.zeros((N, T, H))              # 为h设定维度
    cache = []                           # 指定cache为列表
    h_input = h0                         # (N,H)
    for step in range(T):
        h_output, step_cache = rnn_step_forward(x[:,step,:], h_input, Wx, Wh, b)    # (N,H)
        h[:,step,:] = h_output           # 将当前时刻状态保存 (N,D)
        cache.append(step_cache)         # 将当前时刻的缓存变量保存，用于反向传播
        h_input = h_output               # 将该时刻的状态保存然后送入到下个时刻
    cache.append(D)                      # 将数据维度D添加到cache最后一个位置
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape                                  # 提取梯度的维度
    D   = cache[-1]                     
    dx  = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db  = np.zeros(H)
    # 每一个时刻通过h计算分数，这里的dh是从分数传回来的
    step_dprev_h = np.zeros((N, H))
    for step in list(range(T))[::-1]:                    # 反向传播所以从后往前传播
        dh_upstream = step_dprev_h + dh[:,step,:]        # 从后面时刻传来的step_dprev_h (N,H)               
        step_dx, step_dprev_h, step_dWx, step_dWh, step_db = rnn_step_backward(dh_upstream, cache[step])        
        dx[:,step,:]  = step_dx
        dWx += step_dWx
        dWh += step_dWh
        db  += step_db
    dh0 = step_dprev_h                                   # 获取h0的梯度                  

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # 使用x中的元素挑选权重W的列
    out = W[x, :]                                    # (N,T,D)
    # print(out.shape)      
    cache = (x, W.shape)                                  # (V,D)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass #       x:(N,T)   W:(V,D)

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, (V, D) = cache
    dW = np.zeros((V, D))                      # 创建指定维度的dW
    # print(dout[x].shape)
    # 将每一个 (N,T)位置对应的索引v对应的D维数据叠加到(V,D)中第v维
    np.add.at(dW, x, dout)                     # 每一维度的使用量

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]          # ???怎么保证数值稳定性

    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape                                # (N,H)
    activation = x.dot(Wx) + prev_h.dot(Wh) + b        # (N,4H)
    # 将激活输出按列分为四份 i,f,o,g
    i = sigmoid(activation[:,:H])                      # i,f,o,g四个维度都是 (N,H)
    f = sigmoid(activation[:,H:2*H])
    o = sigmoid(activation[:,2*H:3*H])
    g = np.tanh(activation[:,3*H:])

    next_c = f * prev_c + i * g                        # 计算当前时刻细胞单元Ct
    next_h = o * np.tanh(next_c)                       # 计算当前时刻的状态ht
    cache = (x, prev_h, prev_c, Wx, Wh, i , f, o, g, next_c, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, i , f, o, g, next_c, next_h = cache
    # dnext_h传回来的梯度
    c_out = np.tanh(next_c)                                # 细胞单元的激活输出
    do = dnext_h * c_out                                   # (N,H)
    dnext_c2 = dnext_h*o*(1-c_out*c_out)                   # next_c的第二个梯度 (N,H)

    # dnext_c传回来的梯度
    dnext_c += dnext_c2                                    # 将next_c与next_h回传到next_c的梯度累加
    di = dnext_c * g                                       # (N,H)
    dg = dnext_c * i                                       # (N,H)
    df = dnext_c * prev_c                                  # (N,H)

    dact_i = di*i*(1-i)                                    # (N,H) 注意激活函数 
    dact_f = df*f*(1-f)
    dact_o = do*o*(1-o)
    dact_g = dg*(1-g*g)
    dactivation = np.hstack((dact_i, dact_f, dact_o, dact_g))   # (N,4H) 按列拼接

    dprev_c = dnext_c * f                                       # (N,H)
    dprev_h = dactivation.dot(Wh.T)                             # (N,H)

    # 矩阵求导，activation = Wx，对x求导是上游梯度与W的转置的内积，对哪一边求导
    # 就把与被求导向乘的矩阵取转置放到那一边，剩下一边是上游梯度!!!
    dx = dactivation.dot(Wx.T)                                  # (N,D)
    dWx = x.T.dot(dactivation)                                  # (D,4H)
    dWh = prev_h.T.dot(dactivation)                             # (H,4H)
    db = np.sum(dactivation, axis=0)                            # (4H,)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)        注意D是经过词嵌入后的维度
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape                                    # 获取序列长度T
    H = h0.shape[1]
    prev_h = h0
    prev_c = np.zeros_like(h0)                           # 初始化细胞状态C0为0 (N,H)
    h = np.zeros((N,T,H))                                # 创建隐藏状态向量 (N,T,H)
    cache = []                                           # 定义空列表来存储每个时刻前向传播的cache
    for step in range(T):
        # next_h, next_c的维度都是 (N,H)
        next_h, next_c, step_cache = lstm_step_forward(x[:,step,:], prev_h, prev_c, Wx, Wh, b)
        h[:,step,:] = next_h                             # 存储隐藏状态向量
        cache.append(step_cache)                         # 存储cache值
        prev_h = next_h                                  # 将该时刻的状态送入下个时刻 (N,H)
        prev_c = next_c                                  # (N,H)
    cache.append(D)                                      # 加入嵌入维度，用于反向传播 长度为:T+1
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H) 这是当前时刻从分数传回来的dh梯度
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape
    D = cache[-1]                                     # 获取词嵌入维度D
    dx  = np.zeros((N,T,D))                           # 初始化固定维度的零向量
    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,4*H))
    dWh = np.zeros((H,4*H))
    db  = np.zeros(4*H)

    dnext_c = np.zeros((N,H))                         # 没有外界向细胞单元传入梯度，所以为0
    dlstm_next_h = np.zeros((N,H))                    # 最后时刻的h只有从分数来的梯度
    # 注意：细胞状态梯度只有上一时刻LSTM回传的梯度；但隐藏状态包含分数回传及上一时刻LSTM单元回传的梯度
    # 在最后一个时刻没有从后面传来的梯度，所以dnext_c，dprev_next_h都设为0
    for step in list(range(T))[::-1]:
        dnext_h = dh[:,step,:] + dlstm_next_h         # 分数+后一时刻lstm单元回传梯度
        step_dx, step_dprev_h, step_dprev_c, step_dWx, step_dWh, step_db = lstm_step_backward(dnext_h, dnext_c, cache[step])
        dx[:,step,:] = step_dx                        # (N,D)
        dWx += step_dWx                               # (D,4H)
        dWh += step_dWh                               # (H,4H)
        db  += step_db                                # (4H,)

        dlstm_next_h = step_dprev_h                   # (N,H) 更新对隐藏单元h的梯度
        dnext_c = step_dprev_c                        # (N,H) 更新对细胞单元的梯度
    dh0 = step_dprev_h 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)             # (N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T          # (D, M)
    db = dout.sum(axis=(0, 1))                                        # (M, )

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)                         # reshape为(N*T, V)便于处理
    y_flat = y.reshape(N * T)                            # 使其与x_flat的维度对应
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True)) # 每行减最大值，保持数值稳定性 (N*T, V)
    probs /= np.sum(probs, axis=1, keepdims=True)                  # 按公式只需计算标签部分，这里都计算了
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N   # 只取标签y_i对应的数据

    dx_flat = probs.copy()                                         # (N*T, V)
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]                                  # 添加None使维度对齐

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
