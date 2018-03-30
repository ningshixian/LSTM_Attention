class AttentionLayer(Layer):
    def __init__(self,attended_dim,state_dim,
                source_dim,scoreFunName='Euclidean',
                 atten_activation='tanh',name='AttentionLayer'):
#{{{
        self.attended_dim=attended_dim;
        self.state_dim=state_dim;
        self.source_dim=source_dim;
        self.init=initializations.get('glorot_uniform');
        self.name=name;
        self.one_init=initializations.get('one');
        self.atten_activation=activations.get(atten_activation);
        self.scoreFunName=scoreFunName;
        self.eps=1e-5;
        #self.source_dim=glimpsed_dim;
        super(AttentionLayer,self).__init__();
    #}}}
    def euclideanScore(self,attended,state,W):
#{{{
        #Euclidean distance 
        M=(attended-state)**2;
        M=T.dot(M,W);
        _energy=M.max()-M;
        return _energy; 
#}}}
    def manhattenScore(self,attended,state,W):
#{{{
        #Manhattan Distance 
        #eps for avoid gradient to be NaN;
        M=T.abs_(T.maximum(attended-state,self.eps));
        M=T.dot(M,W);
        _energy=M.max()-M;
        return _energy; 
#}}}
    def bilinearScore(self,attended,state,W):
#{{{
        #Bilinear function  
        M=(attended*state*W).sum(axis=-1);
        _energy=self.atten_activation(M);
        return _energy;
#}}}
    def forwardNNScore(self,attended,state,W):
#{{{
        #get weights
        W_1=W[:(self.attended_dim+self.state_dim)*self.state_dim]; 
        W_1=W_1.reshape((self.attended_dim+self.state_dim,self.state_dim));
        W_2=W[(self.attended_dim+self.state_dim)*self.state_dim:];
        
        #forward neural network 
        state_=T.repeat(state.reshape((1,-1)),attended.shape[0],axis=0);
        input=T.concatenate([attended,state_],axis=-1);
        M1=self.atten_activation(T.dot(input,W_1));
        M2=self.atten_activation(T.dot(M1,W_2));
        _energy=M2;
        return _energy;
    #}}}
    def CNNScore(self,attended,state,W):
#{{{
        state_=T.repeat(state.reshape((1,-1)),attended.shape[0],axis=0);
        input=T.concatenate([attended,state_],axis=-1);
        M1=self.CNN1.call(input);
        M2=self.CNN2.call(M1);
        _energy=M2.flatten();
        return _energy;
#}}}
    def CosineScore(self,attended,state,W):
#{{{
        dotProduct=T.dot(attended,state.T);
        Al2Norm=T.sqrt((attended**2).sum(axis=-1));
        Bl2Norm=T.sqrt((state**2).sum(axis=-1));
        M=dotProduct/(Al2Norm*Bl2Norm);
        _energy=T.exp(M+2);
        return _energy;
#}}}
    def vanilaScore(self,attended,state,W):
        """
            the origin score proprosed by Bahdanau 2015
        """

    def build(self):
#{{{
        self.W_A_X=shared((self.attended_dim,self.attended_dim),
                             name='{}_W_A_X'.format(self.name));
        self.b_A_X=shared((self.attended_dim,),
                            name='{}_W_A_b'.format(self.name));
        self.W_A_h=shared((self.attended_dim,self.attended_dim),
                             name='{}_W_A_h'.format(self.name));
        self.W_A_combine=shared((self.source_dim*2,
                                 self.source_dim),
                               name='{}_W_A_combine'.format(self.name));
        self.b_A_combine=shared((self.source_dim,),
                               name='{}_b_A_combine'.format(self.name))
        #self.W_A_combine=shared((self.source_dim,
        #                         self.source_dim),
        #                         name='{}_W_A_combine'.format(self.name));
        #self.b_A_combine=shared((self.source_dim,),
        #                         name='{}_b_A_combine'.format(self.name))
        #use constraint
        self.constraints={}
        
        self.params=[
                     self.W_A_X,self.b_A_X,
                    # self.W_A_h,
                     self.W_A_combine,self.b_A_combine
                    ];
        
        #for attention weight and score function
        if self.scoreFunName == "Euclidean":
#{{{
            self.W_A=shared((self.state_dim,),
                          name='{}_W_A'.format(self.name));
            self.W_A.set_value(np.ones((self.state_dim,),dtype=theano.config.floatX));
            self.constraints[self.W_A]=self.NonNegConstraint;
            self.scoreFun=self.euclideanScore;
            self.params.append(self.W_A);
#}}}
        elif self.scoreFunName == "Bilinear":
#{{{
            assert self.attended_dim==self.state_dim,"in Bilinear score function,"\
                " attended_dim must be equal to state_dim"
            self.W_A=self.init((self.state_dim,),
                                name="{}_W_A".format(self.name));
            self.scoreFun=self.bilinearScore;
            self.params.append(self.W_A);
#}}}
        elif self.scoreFunName == "forwardNN":
#{{{
            #this is two layer NN 
            #first layer (attended_dim+state_dim,state_dim);
            #second layer (state_dim,1);
            self.W_A=shared(((self.attended_dim+self.state_dim)\
                                *self.state_dim+self.state_dim,),
                                name="{}_W_A".format(self.name));
            self.scoreFun=self.forwardNNScore;
            self.params.append(self.W_A);
#}}}
        elif self.scoreFunName == "CNN":
#{{{
            #this if one layer CNN and pool layer;
            nb_filter=(self.attended_dim+self.state_dim)/2;
            filter_length=3;
            input_dim=self.attended_dim+self.state_dim;
            self.CNN1=Convolution1D(nb_filter=nb_filter,
                                   filter_length=filter_length,
                                  input_dim=input_dim,activation='tanh',
                                  border_mode='same');
            self.CNN2=Convolution1D(nb_filter=1,
                                   filter_length=filter_length,
                                  input_dim=nb_filter,activation='tanh',
                                  border_mode='same');
            self.W_A=self.CNN1.W;
            self.scoreFun=self.CNNScore;
            self.params.append(self.W_A);
            self.params.append(self.CNN2.W);
#}}}
        elif self.scoreFunName == "Cosine":
#{{{
            self.scoreFun=self.CosineScore;
            self.W_A=None;
#}}}
        elif self.scoreFunName == "Manhatten":
#{{{
            self.scoreFun=self.manhattenScore;
            self.W_A=self.one_init((self.state_dim,),
                          name='{}_W_A'.format(self.name));
            self.constraints[self.W_A]=self.NonNegConstraint;
            self.params.append(self.W_A);
#}}}
        else:
            assert 0, "we only have Euclidean, Bilinear, forwardNN"\
                    " score function for attention";

#}}}
    def softmaxReScale(self,energy_,threshould):
#{{{
        #in energy_, the goundthrud should be max
        assert energy_.ndim==1;
        #convert threshould from percentage to energy_;
        threshould_=T.log(T.exp(energy_-energy_.max()).sum())+T.log(threshould)+energy_.max()
        energy=self.reScale(energy_,threshould_);
        return T.nnet.softmax(energy);
    #}}}
    def reScale(self,energy,threshold,replaceValue=1e-7):
#{{{
        assert energy.ndim==1;
        maxValue=energy.max();
        def checkThreshold(value,threshold,replaceValue):
            return T.switch(T.lt(value,threshold),replaceValue,value);
        result,update=theano.scan(fn=checkThreshold,
                                 outputs_info=None,
                                 sequences=[energy],
                                 non_sequences=[threshold,replaceValue]);
        return T.switch(T.lt(maxValue,threshold),energy,result);
#}}}
    
    def step(self,state,attended,source):
        #from theano.gradient import disconnected_grad;
        #state=disconnected_grad(state_);
        #M_state=T.dot(self.W_A_h,state) ;

        _energy=self.scoreFun(attended,state,self.W_A)
        energy=T.nnet.softmax(_energy);
        #energy=self.softmaxReScale(_energy,0.02);
        #energy=self.reScale(energy.flatten(),0.02).reshape((1,-1))
        #energyIndex=energy.flatten().argmin(axis=-1);
        glimpsed=(energy.T*source).sum(axis=0)
        #glimpsed=source[energyIndex];
        return energy.flatten(),glimpsed;

    def NonNegConstraint(self,p):
        p*=K.cast(p>=0.,K.floatx());
        return p;

    def link(self,attended,state,source):
        step_function=self.step;
        attended_=T.tanh(T.dot(attended,self.W_A_X))+self.b_A_X;
        #attended_=attended;
        [energy,glimpsed],_=theano.scan(fn=step_function,
                            sequences=[attended_],
                               outputs_info=None,
                            non_sequences=[attended_,source]);
        self.energy=energy;
        
        #combine 
        #combine=T.concatenate([glimpsed,attended],axis=-1);
        combine=T.concatenate([glimpsed,source],axis=-1);
        combined=T.tanh(T.dot(combine,self.W_A_combine))+self.b_A_combine;
        #no source
        #combined=T.tanh(T.dot(glimpsed,self.W_A_combine))+self.b_A_combine;
        return combined;