#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from keras.layers import activations, Wrapper
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LSTM


class AttentionLSTM(LSTM):
    def __init__(self, output_dim, attention_vec, **kwargs):
        self.attention_vec = attention_vec
        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        '''
        this method initializes all of the weight matrices we need for the attentional component
        :param input_shape:
        :return:
        '''
        super(AttentionLSTM, self).build(input_shape)

        assert hasattr(self.attention_vec, '_keras_shape')
        attention_dim = self.attention_vec._keras_shape[1]

        self.U_a = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.inner_init((attention_dim, self.output_dim),
                                   name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

        self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_s'.format(self.name))
        self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.U_m, self.U_s,
                                   self.b_a, self.b_m, self.b_s]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        '''
        This method is used by the RNN superclass, and tells the function what to do on each timestep.
        :param x:
        :param states:
        :return:
        '''
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]

        m = K.tanh(K.dot(h, self.U_a) + attention + self.b_a)
        s = K.exp(K.dot(m, self.U_s) + self.b_s)
        h = h * s

        return h, [h, c]

    def get_constants(self, x):
        '''
        This method is used by the LSTM superclass to define components outside of the step function,
        so that they don’t need to be recomputed every time step.
        :param x:
        :return:
        '''
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants


class AttentionLSTM_t(LSTM):
    def __init__(self, output_dim, attn_activation='tanh', **kwargs):
        self.attn_activation = activations.get(attn_activation)
        super(AttentionLSTM_t, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM_t, self).build(input_shape)

        # assert hasattr(self.attention_vec, '_keras_shape')
        # attention_dim = self.attention_vec._keras_shape[1]

        self.U_a = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_s'.format(self.name))
        self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.b_a, self.U_s, self.b_s]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        '''
        This method is used by the RNN superclass, and tells the function what to do on each timestep.
        :param x:
        :param states:
        :return:
        '''
        h, [h, c] = super(AttentionLSTM_t, self).step(x, states)

        m = K.tanh(K.dot(h, self.U_a) + self.b_a)
        alpha = K.exp(K.dot(m, self.U_s) + self.b_s)
        h = h * alpha

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM_t, self).get_constants(x)
        return constants


class AttentionLSTMWrapper(Wrapper):
    def __init__(self, layer, attn_activation='tanh', single_attention_param=False, **kwargs):
        assert isinstance(layer, LSTM)
        self.supports_masking = True
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param
        super(AttentionLSTMWrapper, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(AttentionLSTMWrapper, self).build()

        self.U_a = self.layer.inner_init((self.layer.output_dim, self.layer.output_dim),
                                         name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.layer.output_dim,), name='{}_b_a'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.layer.inner_init((self.layer.output_dim, 1), name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.layer.inner_init((self.layer.output_dim, self.layer.output_dim),
                                             name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.layer.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights = [self.U_a, self.U_s, self.b_a, self.b_s]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        h, [h, c] = self.layer.step(x, states)
        m = self.attn_activation(h)
        s = K.softmax(K.dot(m, self.U_s))
        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.layer.output_dim, axis=1)
        else:
            h = h * s
        # attention = states[4]
        #
        # m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        # s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)
        #
        # if self.single_attention_param:
        # h = h * K.repeat_elements(s, self.layer.output_dim, axis=1)
        # else:
        #     h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output


