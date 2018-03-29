# LSTM_Attention

GitHub 项目

[datalogue/keras-attention](https://github.com/datalogue/keras-attention/blob/master/models/custom_recurrents.py)

博客

[How to Develop an Encoder-Decoder Model with Attention for Sequence-to-Sequence Prediction in Keras](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)

![1](1.png)

###步骤解读

![](https://cdn-images-1.medium.com/max/1200/1*EKRQPw2bbthCrM2ROEJr5A.png)

**Equation 1: **利用一个前馈神经网络计算在预测字符 t时字符 *j* 的重要性；**Equation 2:** softmax 函数归一化

![](https://cdn-images-1.medium.com/max/1200/1*59ngdYa-ROf5p8w1_r0OdQ.png)

**Equation 3: **计算解码器隐层字符 t 的上下文向量（*context vector* ）.

![](https://cdn-images-1.medium.com/max/1200/1*kHUNMl5vCvMu4MjyxE-sfw.png)

**Equation 4: **Reset gate. **Equation 5:** Update gate. **Equation 6: **Proposal hidden state. **Equation 7: **New hidden state.

![](https://cdn-images-1.medium.com/max/1200/1*zC7qzkzIAX8YgSoIVQWK1w.png)

**Equation 8:** A simple neural network to predict the next character.



####理解了公式后，一步一步讲解code

一个简单的 minimal custom Keras layer 需要实现以下方法：

- `__init__`
-  `compute_ouput_shape`
-  `build` 
-  `call`

若要实现复杂功能，需要实现以下方法：

- `get_config`: which allows you to load the model back into memory easily. 

除此之外，若要实现 a Keras recurrent layer， 还需要：

-  `step`



这些方法的功能？

- `__init__` 初始化参数设置+参数传递。如：`self.return_sequences=True`.
- `build` 这是定义权重的方法， 在`Model.compile(…)`时被调用。可训练的权重应该在这里被加入列表``self.trainable_weights`中。
- `call(x)`：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心`call`的第一个参数：输入张量。The `_time_distributed_dense` 函数计算 Equation 1 的前一项。
- `compute_output_shape(input_shape)`：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断
- `step`：代码最重要的部分，主要是完成循环的 cell 的逻辑。Recall that `step` is applied to every element of the input sequence.





### References

Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. ["Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).](https://arxiv.org/abs/1409.0473)