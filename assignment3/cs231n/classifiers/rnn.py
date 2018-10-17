from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type                                   # 获取输入参数
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}    # 序号到单词的字典
        self.params = {}

        vocab_size = len(word_to_idx)                                # 获取字典大小

        self._null = word_to_idx['<NULL>']                           # 表示字典中的特殊token值
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)  # (V,W)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters 将图像维度转换为h0的维度
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)     # (D,H)
        self.params['W_proj'] /= np.sqrt(input_dim)                        # 消除方差的差异
        self.params['b_proj'] = np.zeros(hidden_dim)                       # 偏置全初始为0

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)  # (W, dim_mul*H)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)   # (H, dim_mul*H)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)        # (H,V)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)                               # 规范参数类型


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]                   # 除去最后一个单词 (N, T-1) ??与上述描述不对应
        captions_out = captions[:, 1:]                   # 除去第一个单词   (N, T-1)
        # print('---------------')
        # list_str1 = [self.idx_to_word[i] for i in captions[0]]
        # print('[total]:', ' '.join(list_str1))
        # list_str2 = [self.idx_to_word[i] for i in captions_in[0]]
        # print('[input]:', ' '.join(list_str2))
        # list_str3 = [self.idx_to_word[i] for i in captions_out[0]]
        # print('[output]:', ' '.join(list_str3))
        # print('---------------')
        # You'll need this
        mask = (captions_out != self._null)              # 获取标志位表示非'<NULL>' (N, T-1)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        #                                                                          #
        # Note also that you are allowed to make use of functions from layers.py   #
        # in your implementation, if needed.                                       #
        ############################################################################
        # ------------------(1) 计算损失--------------------------
        # 1.将视觉特征向量转换为RNN的h0状态
        h0_state = features.dot(W_proj) + b_proj   # (N, H)
        # 2.将单词进行嵌入，增加W维度，训练模型时<START> <END>已经在captions变量里了!!!
        # 所有这里 captions_in/captions_out 变为T-1维度
        embed_out, embed_cache = word_embedding_forward(captions_in, W_embed) # (N, T-1) => (N,T-1,W)
        # 3.RNN按时间进行前向传播
        rnn_out, rnn_cache = rnn_forward(embed_out, h0_state, Wx, Wh, b)      # (N,T-1,H)
        # 4.计算每个时刻的输出分数
        affine_out, affine_cache = temporal_affine_forward(rnn_out, W_vocab, b_vocab) # (N,T-1,V)
        # 5.计算每个时刻的损失 梯度：(N, T-1, V) 损失：一个值
        loss, softmax_grad = temporal_softmax_loss(affine_out, captions_out, mask, verbose=False)     # 一个值
       
        # --------------------(2) 计算梯度---------------------------
        # 4.计算分数的梯度 回传的梯度：(N,T-1,V) => (N,T-1,H)
        grad_upstream, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(softmax_grad, affine_cache)        
        # 3.计算RNN的反向传播 回传的梯度：(N,T-1,H) => (N,T-1,W)
        grad_upstream, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(grad_upstream, rnn_cache)       
        # 2.计算词嵌入的梯度 嵌入矩阵梯度：(N,T-1,W) => (V,W)
        grads['W_embed'] = word_embedding_backward(grad_upstream, embed_cache)
        # 1.计算视觉特征的转换矩阵的梯度 dh0: (N,H)
        grads['W_proj'] = features.T.dot(dh0)                # (D,H)
        grads['b_proj'] = np.sum(dh0, axis=0)                # (H,)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of
         shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        h0_state = features.dot(W_proj) + b_proj                   # 通过视觉特征得到h0状态 (N,H)
        prev_h = h0_state                                          # 首先将h0状态送到rnn中 
        sample_word = np.ones((N,1), dtype=np.int32)*self._start   # 将<START>作为第一个单词输入 (N,1) 
        for step in range(max_length):
            embed_word, _ = word_embedding_forward(sample_word, W_embed)  # 进行词嵌入(N,1,W)
            # print('embed1:', embed_word.shape)
            embed_word = embed_word.squeeze()                             # (N,W)
            # print('embed2:', embed_word.shape)
            h_out, _ = rnn_step_forward(embed_word, prev_h, Wx, Wh, b)    # RNN前向传播 h: (N,H)
            scores = h_out.dot(W_vocab) + b_vocab                         # 计算分数 (N,V)
            
            sample_word = np.argmax(scores, axis=1)                       # (N,)
            captions[:,step] = sample_word                                # 将采用的单词保存到captions中
            sample_word = sample_word.reshape(-1,1) 
            prev_h = h_out                                                # 将当前的状态送入下个时刻

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
