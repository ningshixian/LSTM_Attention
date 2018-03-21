'''
https://github.com/roebius/deeplearning_keras2/blob/master/nbs2/attention_wrapper.py
'''

# - Introduced a few changes for Keras 2
from keras import backend as K
# from keras.initializations import zero  # - Not used with Keras 2
from keras.engine import InputSpec
from keras.models import Sequential
from keras.layers import LSTM, activations, Wrapper, Recurrent, Layer

class Attention(Layer):
    def __init__(self, fn_rnn, nlayers=1, **kwargs):
        self.supports_masking = True
        self.fn_rnn = fn_rnn
        self.nlayers = nlayers
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        super(Attention, self).__init__(**kwargs)


    def all_attrs(self, name):
        return sum([getattr(layer, name, []) for layer in self.layers], [])


    def w(self, dims, init, name):
        # - Keras 2 different parameter order
        return self.add_weight(dims, name=name.format(self.name), initializer=init) 


    def build(self, input_shape):
        self.enc_shape = input_shape
        assert len(self.enc_shape) >= 3
        self.layers = [self.fn_rnn for i in range(self.nlayers)]
        nb_samples, nb_time, nb_dims = self.enc_shape
        l0 = self.layers[0]

        out_shape = self.compute_output_shape(input_shape)  # - changed from self.get_output_shape_for
        for layer in self.layers:
            if not layer.built: layer.build(out_shape)

        init = l0.kernel_initializer  # - changed from l0.init
        out_dim = l0.units  # - changed from l0.output_dim
        self.W1 = self.w((self.enc_shape[-1], nb_dims), init, '{}_W1')
        self.W2 = self.w((out_dim, nb_dims), init, '{}_W2')
        self.b2 = self.w((nb_dims,), 'zeros', '{}_b2') # - changed from zero
        self.V =  self.w((nb_dims,), init, '{}_V')
        self.W3 = self.w((nb_dims+out_dim, out_dim), init, '{}_W3')
        self.b3 = self.w((out_dim,), 'zeros', '{}_b3') # - changed from zero

        self.trainable_weights += self.all_attrs( 'trainable_weights')
        self.non_trainable_weights += self.all_attrs( 'non_trainable_weights')
        #self.losses += self.all_attrs( 'losses')  # - seems not available for layer in Keras 2
        #self.updates = self.all_attrs( 'updates') # - seems not available for layer in Keras 2
        self.constraints = getattr(self.layers[0], 'constraints', {}) # FIXME
        super(Attention, self).build(input_shape)

        
    def compute_output_shape(self, input_shape): # - changed from self.get_output_shape_for
        # return (input_shape[0],input_shape[1],input_shape[2])
        return self.layers[0].compute_output_shape(input_shape)  # - changed from self.get_output_shape_for

    def step(self, x, states):
        h = states[0]
        enc_output = states[-1]
        xW1 = states[-2]

        hW2 = K.expand_dims(K.dot(h,self.W2)+self.b2, 1)
        u = K.tanh(xW1+hW2)
        a = K.expand_dims(K.softmax(K.sum(self.V*u,2)), -1)
        Xa = K.sum(a*enc_output,1)
        h = K.dot(K.concatenate([x,Xa],1),self.W3)+self.b3

        for layer in self.layers: h, new_states = layer.step(h, states)
        return h, new_states


    def get_constants(self, enc_output, constants):
        constants.append(K.dot(enc_output,self.W1))
        constants.append(enc_output)
        return constants


    def compute_mask(self, input, mask):
        return self.layers[0].compute_mask(input, mask[1])


    def call(self, x, mask=None):
        l0 = self.layers[0]
        enc_output = x
        # enc_output, dec_input = x

        if l0.stateful: initial_states = l0.states
        else: initial_states = l0.get_initial_state(dec_input)  # - changed from get_initial_states
        constants = l0.get_constants(dec_input)
        constants = self.get_constants(enc_output, constants)
        preprocessed_input = l0.preprocess_input(dec_input)
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
             initial_states, go_backwards=l0.go_backwards,  mask=mask, # - changed from mask=mask[1]
             constants=constants, unroll=l0.unroll, input_length=self.dec_shape[1])
        if l0.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((l0.states[i], states[i]))

        return outputs if l0.return_sequences else last_output

