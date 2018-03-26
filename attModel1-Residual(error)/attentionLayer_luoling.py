
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, activations
from keras.layers.wrappers import TimeDistributed
from keras.engine import InputSpec
from keras.layers.recurrent import Recurrent
# import theano as th


class AttentionLayer(Layer):
    def __init__(self, attended_dim, state_dim,
                 source_dim, scoreFunName='Euclidean',
                 atten_activation='tanh', name='AttentionLayer'):
        # {{{
        self.attended_dim = attended_dim
        self.state_dim = state_dim
        self.source_dim = source_dim
        self.init = initializers.get('glorot_uniform')
        self.name = name
        self.one_init = initializers.get('one')
        self.atten_activation = activations.get(atten_activation)
        self.scoreFunName = scoreFunName
        self.eps = 1e-5
        # self.source_dim=glimpsed_dim
        super(AttentionLayer, self).__init__()

    # }}}
    def euclideanScore(self, attended, state, W):
        # {{{
        # Euclidean distance
        M = (attended - state) ** 2
        M = K.dot(M, W)
        _energy = M.max() - M
        return _energy

    # }}}
    def manhattenScore(self, attended, state, W):
        # {{{
        # Manhattan Distance
        # eps for avoid gradient to be NaN
        M = K.abs(K.maximum(attended - state, self.eps))
        M = K.dot(M, W)
        _energy = M.max() - M
        return _energy

    # }}}
    def bilinearScore(self, attended, state, W):
        # {{{
        # Bilinear function
        M = (attended * state * W).sum(axis=-1)
        _energy = self.atten_activation(M)
        return _energy

    # }}}
    def forwardNNScore(self, attended, state, W):
        # {{{
        # get weights
        W_1 = W[:(self.attended_dim + self.state_dim) * self.state_dim]
        W_1 = W_1.reshape((self.attended_dim + self.state_dim, self.state_dim))
        W_2 = W[(self.attended_dim + self.state_dim) * self.state_dim:]

        # forward neural network
        state_ = K.repeat(state.reshape((1, -1)), attended.shape[0], axis=0)
        input = K.concatenate([attended, state_], axis=-1)
        M1 = self.atten_activation(K.dot(input, W_1))
        M2 = self.atten_activation(K.dot(M1, W_2))
        _energy = M2
        return _energy

    # }}}
    def CosineScore(self, attended, state, W):
        # {{{
        dotProduct = K.dot(attended, state.K)
        Al2Norm = K.sqrt((attended ** 2).sum(axis=-1))
        Bl2Norm = K.sqrt((state ** 2).sum(axis=-1))
        M = dotProduct / (Al2Norm * Bl2Norm)
        _energy = K.exp(M + 2)
        return _energy

    # }}}
    def vanilaScore(self, attended, state, W):
        """
            the origin score proprosed by Bahdanau 2015
        """


    def build(self, input_shape):
        # {{{
        assert len(input_shape) == 3

        self.W_A_X = self.add_weight((self.attended_dim, self.attended_dim, ),
                                     initializer=self.init,
                                    name='{}_W_A_X'.format(self.name))
        self.b_A_X = self.add_weight((self.attended_dim,),
                                     initializer='zero',
                                    name='{}_W_A_b'.format(self.name))
        self.W_A_h = self.add_weight((self.attended_dim, self.attended_dim),
                                     initializer=self.init,
                                    name='{}_W_A_h'.format(self.name))
        self.W_A_combine = self.add_weight((self.source_dim * 2, self.source_dim),
                                           initializer=self.init,
                                            name='{}_W_A_combine'.format(self.name))
        self.b_A_combine = self.add_weight((self.source_dim,),
                                           initializer='zero',
                                            name='{}_b_A_combine'.format(self.name))
        # self.W_A_combine=shared((self.source_dim,
        #                         self.source_dim),
        #                         name='{}_W_A_combine'.format(self.name))
        # self.b_A_combine=shared((self.source_dim,),
        #                         name='{}_b_A_combine'.format(self.name))
        # use constraint
        self.constraints = {}

        self.params = [
            self.W_A_X, self.b_A_X,
            # self.W_A_h,
            self.W_A_combine, self.b_A_combine
        ]

        # for attention weight and score function
        if self.scoreFunName == "Euclidean":
            # {{{
            self.W_A = self.add_weight((self.state_dim,),
                                       initializer='one',
                                        name='{}_W_A'.format(self.name))

            # self.W_A = K.ones((self.state_dim,), name='{}_W_A'.format(self.name))
            # self.weights.append(self.W_A)

            self.constraints[self.W_A] = self.NonNegConstraint
            self.scoreFun = self.euclideanScore
            self.params.append(self.W_A)
        # }}}
        elif self.scoreFunName == "Bilinear":
            # {{{
            assert self.attended_dim == self.state_dim, "in Bilinear score function," \
                                                        " attended_dim must be equal to state_dim"
            self.W_A = self.init((self.state_dim,),
                                    name="{}_W_A".format(self.name))
            self.scoreFun = self.bilinearScore
            self.params.append(self.W_A)
        # }}}
        elif self.scoreFunName == "forwardNN":
            # {{{
            # this is two layer NN
            # first layer (attended_dim+state_dim,state_dim)
            # second layer (state_dim,1)
            self.W_A = self.add_weight(((self.attended_dim + self.state_dim) \
                                        * self.state_dim + self.state_dim,),
                              name="{}_W_A".format(self.name))
            self.scoreFun = self.forwardNNScore
            self.params.append(self.W_A)
        # }}}
        elif self.scoreFunName == "Cosine":
            # {{{
            self.scoreFun = self.CosineScore
            self.W_A = None
        # }}}
        elif self.scoreFunName == "Manhatten":
            # {{{
            self.scoreFun = self.manhattenScore
            self.W_A = self.one_init((self.state_dim,),
                                     name='{}_W_A'.format(self.name))
            self.constraints[self.W_A] = self.NonNegConstraint
            self.params.append(self.W_A)
        # }}}
        else:
            assert 0, "we only have Euclidean, Bilinear, forwardNN" \
                      " score function for attention"

        # self.build = True
        super(AttentionLayer, self).build(input_shape)


    def step(self, state, attended, source):
        # from theano.gradient import disconnected_grad
        # state=disconnected_grad(state_)
        # M_state=K.dot(self.W_A_h,state)

        _energy = self.scoreFun(attended, state, self.W_A)
        energy = K.softmax(_energy)
        # energy=self.softmaxReScale(_energy,0.02)
        # energy=self.reScale(energy.flatten(),0.02).reshape((1,-1))
        # energyIndex=energy.flatten().argmin(axis=-1)
        glimpsed = (energy.K * source).sum(axis=0)
        # glimpsed=source[energyIndex]
        return energy.flatten(), glimpsed

    def NonNegConstraint(self, p):
        p *= K.cast(p >= 0., K.floatx())
        return p

    def call(self, inputs, mask=None):
        attended = inputs[0]
        state = inputs[1]
        source = inputs[2]
        step_function = self.step
        attended_ = K.tanh(K.dot(attended, self.W_A_X)) + self.b_A_X
        # attended_=attended
        [energy, glimpsed], _ = step_function(state, attended_, source)
        # [energy, glimpsed], _ = th.scan(fn=step_function,
        #                                     sequences=[attended_],
        #                                     outputs_info=None,
        #                                     non_sequences=[attended_, source])
        self.energy = energy

        # Z = tanh(W[g;h])
        combine = K.concatenate([glimpsed, source], axis=-1)
        combined = K.tanh(K.dot(combine, self.W_A_combine)) + self.b_A_combine
        # no source: Z = tanh(W[g])
        # combined=T.tanh(T.dot(glimpsed,self.W_A_combine))+self.b_A_combine
        return combined


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[-1])
