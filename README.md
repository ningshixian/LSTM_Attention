# LSTM_Attention

attention_lstm.py 文件细节

'''
    X = Input Sequence of length n.
    H = LSTM(X); Note that here the LSTM has return_sequences = True,
        so H is a sequence of vectors of length n.
    s is the hidden state of the LSTM (h and c)

    h is a weighted sum over H: 加权和
    h = sigma(j = 0 to n-1)  alpha(j) * H(j)

    weight alpha[i, j] for each hj is computed as follows:
    H = [h1,h2,...,hn]
    M = tanh(H)
    alhpa = softmax(w.transpose * M)

    h# = tanh(h)
    y = softmax(W * h# + b)

    J(theta) = negative_log_likelihood + regularity
'''

参考

[Keras Attention Mechanism](https://github.com/philipperemy/keras-attention-mechanism)
[keras-language-modeling](https://github.com/codekansas/keras-language-modeling/blob/master/keras_models.py)