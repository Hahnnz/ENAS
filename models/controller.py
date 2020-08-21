import os
import numpy as np
import tensorflow as tf
from .layers import *

from .utils import set_optimizer

class Controller():
    def __init__(self, 
                 lr_options,
                 grad_options,
                 batch_size=None, 
                 utype='lstm', # recurrent unit type
                 optimizer_type='adam', # optimizer type
                 num_stack_rnn=1, # Number of Recurrent unit layer
                 unit_dim=32, # number of channel of recurrent unit
                 num_branch=5, # Number of operation (size of search space for operator)
                 num_cells=5,
                 lr_init=1e-3,
                 bl_dec=0.999,
                 entropy_weight=None,
                 op_tanh_reduce=1.0,
                 temperature=None,
                 tanh_constant=None,
                 seed=None, # set random seed of total controller functions
                 graph=None,
                 name='Controller'):
        '''
        Description :
            Create Micro Controller
            
        Attribute Functions :
            [F] __init_controller
            [F] __create_idx_sampler
            [F] __create_op_sampler
            [F] __create_trainer
        '''
        
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type # optimizer type
        self.unit_type = utype # recurrent unit type
        self.num_stack_rnn = num_stack_rnn # Number of Recurrent cell 
        self.unit_dim = unit_dim # number of channel of recurrent unit
        self.num_branch = num_branch # number of hyperparameter you want
        self.num_cells = num_cells #  
        self.lr_init = lr_init
        self.lr_options = lr_options
        self.grad_options = grad_options
        self.entropy_weight = entropy_weight # entropy weight of reward entropy
        self.op_tanh_reduce = op_tanh_reduce # softmax
        self.temperature = temperature # Softmax temperature
        self.tanh_constant = tanh_constant
        self.name = name # Contoller names
        self.bl_dec = bl_dec

        with tf.variable_scope(self.name):
            # Global controller train step
            self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")

            # initialize Controller Recurrent unit weights
            self.__init_controller()

            # Normal Cells
            arc_seq_1, entropy_1, log_prob_1, c, h = self.__create_controller_body(use_bias=True, name='Normal_Cells')
            # Reduce Cells
            arc_seq_2, entropy_2, log_prob_2, _, _ = self.__create_controller_body(prev_c=c, prev_h=h, name='Reduction_Cells')
            self.sample_arc = (arc_seq_1, arc_seq_2)
            self.sample_entropy = entropy_1 + entropy_2
            self.sample_log_prob = log_prob_1 + log_prob_2
            
            self.__create_trainer()
            
    def __init_controller(self):
        '''
        Description :
            initialize necessary weights of controller
        Argument :
            - None
        Output : 
            - None
        '''
        # create recurrent unit layer with initializing 
        self.w_recurrent = []
        for l_id in range(self.num_stack_rnn):
            with tf.variable_scope(f'{self.unit_type}_layer_{l_id}'):
                __num_w_mat = 4 if self.unit_type == 'lstm' else 3
                w = tf.get_variable('weights', [2*self.unit_dim , __num_w_mat*self.unit_dim])
                self.w_recurrent.append(w)

        # create generator embeddings for each branch.
        self.start_vector = tf.get_variable('start_vector', [1, self.unit_dim])
        with tf.variable_scope('search_space_embedding'):
            self.branch_vector = tf.get_variable('branch_vector', [self.num_branch, self.unit_dim])
        '''
        # Softmax weights
        with tf.variable_scope('softmax'):
            self.w_soft  = tf.get_variable('weights', [self.unit_dim, self.num_branch])

            b_init = np.array([10.0, 10.0] + [0] * (self.num_branch - 2), dtype=np.float32)
            self.b_soft = tf.get_variable('bias', [1, self.num_branch], initializer=tf.constant_initializer(b_init))

            # for the case if you dont use softmax weight
            b_soft_no_learn = np.array([0.25, 0.25] + [-0.25] * (self.num_branch - 2), dtype=np.float32)
            b_soft_no_learn = np.reshape(b_soft_no_learn, [1, self.num_branch])
            self.b_soft_no_learn = tf.constant(b_soft_no_learn, dtype=tf.float32)
        '''
    
    def __create_controller_body(self, prev_c=None, prev_h=None, use_bias=False, name='Controller_Body'):
        '''
        Description : 
        Arguments : 
        Outputs :
        '''
        
        def check_length(layer_id, *args):
            '''
            Description :
                check current layer_id. True if layer_id is over then number of child network layer else False.
            Arguments :
                - layer_id [required : tensor integer]: 
                    current child layer (node) id
            Outputs (Boolean):
                True : if current layer id **over** than given number of layers.
                False : if current layer id **less** than given number of layers.
            '''
            import sys
            return tf.less(layer_id, self.num_cells + 2)
        
        # create anchor
        with tf.variable_scope(name):
            anchors = tf.TensorArray(tf.float32, size=self.num_cells + 2, clear_after_read=False, name='Anchor')
            anchors_w1 = tf.TensorArray(tf.float32, size=self.num_cells + 2, clear_after_read=False, name='Anchor_w1')
            # child model Architecture sequence
            arc_seq = tf.TensorArray(tf.int32, size=self.num_cells * 4, name='Architecture_Sequence')
            
            self.__use_bias=use_bias
            # create first layer vector
            prev_c, prev_h, anchors, anchors_w1 = self.__create_anchor_sampler(0, anchors, anchors_w1, prev_c=prev_c, prev_h=prev_h, gen_anchor=2)
            inputs = self.start_vector
            
            def pipeline(layer_id, inputs, prev_c, prev_h, anchors, anchors_w1, arc_seq, entropy, log_prob):
                # node inputs
                inputs, prev_c, prev_h, arc_seq, entropy, log_prob = self.__create_input_sampler(layer_id, inputs, prev_c, prev_h, anchors, anchors_w1, arc_seq, entropy, log_prob)
                # node operations
                inputs, prev_c, prev_h, arc_seq, entropy, log_prob = self.__create_op_sampler(inputs, prev_c, prev_h, arc_seq, entropy, log_prob)
                # node anchor
                prev_c, prev_h, anchors, anchors_w1 = self.__create_anchor_sampler(layer_id, anchors, anchors_w1, inputs=inputs,prev_c=prev_c, prev_h=prev_h, use_bias=self.__use_bias)
                # initialize input as star=t vector
                inputs = self.start_vector
                return (layer_id+1, inputs, prev_c, prev_h, anchors, anchors_w1, arc_seq, entropy, log_prob)

            # loop variables the 
            loop_vars = [
                tf.constant(2, dtype=tf.int32, name='layer_id'), # 0 and 1 are stem layers.
                inputs, prev_c, prev_h,
                anchors, anchors_w1, arc_seq, 
                tf.constant([0.,], dtype=tf.float32, name='entropy'),
                tf.constant([0.,], dtype=tf.float32, name='log_prob'),
            ]

            # Controller rnn cells until layer id reach the number of cells.
            loop_outputs = tf.while_loop(check_length, pipeline, loop_vars, parallel_iterations=1)
            
            _, _, last_c, last_h, _, _, arc_seq, entropy, log_prob = loop_outputs
            
            arc_seq = tf.reshape(arc_seq.stack(), [-1]) # make 1D tensor array
            entropy = tf.reduce_sum(entropy)
            log_prob = tf.reduce_sum(log_prob)
            
            
            
            
        return arc_seq, entropy, log_prob, last_c, last_h
        
    def __create_anchor_sampler(self, layer_id, anchors, anchors_w1, inputs=None, gen_anchor=1, prev_c=None, prev_h=None, use_bias=False):
        '''
        Description :
            create sampler for skip connection, anchor.
        Argument :
            - prev_h (default = None) [required : tensor vecotr, 2D] : 
                hidden states vector of previous RNN units.
            - prev_c (default = None) [required : tensor vector, 2D] : 
                cell states vector of previous RNN units.
            - use_bias (default = False) [required : Boolean]:
                allow to add bias
            - gen_anchor (default = 1) [required : integer]:
                the times you want to generate anchor idx.
        Output : 
            - None
        '''
        if prev_c is None :
            assert prev_h is None, "prev_c and prev_h must both be None or given"
            # for first input of rnn unit. c and h will be zero-vector as same with rnn unit_dim
            create_zero_vec = True
            prev_c = [tf.zeros([1,self.unit_dim], tf.float32) for _ in range(self.num_stack_rnn)]
            prev_h = [tf.zeros([1,self.unit_dim], tf.float32) for _ in range(self.num_stack_rnn)]
        else : 
            create_zero_vec = False
        
        if inputs is None :
            # initialize Generator embedding vectors
            inputs = self.start_vector
            
        for _ in range(gen_anchor):
            next_c, next_h = rnn_layer(inputs, prev_c, prev_h, self.w_recurrent, unit_type='lstm')
            
            attn_w1 = tf.layers.dense(inputs=next_h[-1], units=self.unit_dim, use_bias=False, name='anchor_attn', reuse=tf.AUTO_REUSE)
            anchors_w1 = anchors_w1.write(layer_id, attn_w1)
            
            if create_zero_vec:
                anchors = anchors.write(layer_id, tf.zeros_like(next_h[-1]))
            else :
                anchors = anchors.write(layer_id, next_h[-1])
            
            if gen_anchor > 1:
                layer_id+=1
            
        return next_c, next_h, anchors, anchors_w1
            
    def __create_input_sampler(self, layer_id, data, prev_c, prev_h, anchors, anchors_w1, arc_seq, entropy, log_prob):
        indices = tf.range(0,layer_id, dtype=tf.int32)
        self.__start_id = 4*(layer_id-2) # RNN cell ID
        
        prev_layers = []
        
        # Query input layer idx for a node
        for i in range(2):
            # stack 2 layers for given layer ID.
            # this creates rnn layer that control a specific model layer only
            next_c, next_h = rnn_layer(data, prev_c, prev_h, self.w_recurrent)
            prev_c, prev_h = next_c, next_h
            
            # Query what layer index will be used as node input.
            query = anchors_w1.gather(indices)
            query = tf.reshape(query, [layer_id, self.unit_dim])
            attn_h = tf.layers.dense(inputs=next_h[-1], units=self.unit_dim, use_bias=False, name='vec_attn', reuse=tf.AUTO_REUSE)
            query = tf.tanh(query+attn_h)
            query = tf.layers.dense(inputs=query, units=1, use_bias=False, name='choose_idx', reuse=tf.AUTO_REUSE)
            logits = tf.reshape(query, [1, layer_id])
            
            # Softmax temperature :
            # NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING 
            # https://arxiv.org/pdf/1611.09940.pdf
            # 
            # temperature is hyperparameter set to T = 1 during training. When T > 1, the distribution
            # represented by A(ref, q) becomes less steep, hence preventing the model from being overconfident.
            if self.temperature is not None : 
                logits /= self.temperature
            
            # Logit clipping:
            # NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING 
            # https://arxiv.org/pdf/1611.09940.pdf
            # 
            # tanh_constant is a hyperparameter that controls the range of the logits and hence the entropy of A(ref, q).
            if self.tanh_constant is not None : 
                logits = self.tanh_constant * tf.tanh(logits)
                
            # get best logit and select best layer index to use
            # multinomial == argmax <- not exactly same but do similar work: extract best index among logits.
            # argmax returns a value, EX) 3
            # multinomial returns the probability of class. EX) [0,0,0,1,0]
            index = tf.multinomial(logits, 1)
            index = tf.to_int32(index)
            index = tf.reshape(index, [1])
            
            # write selected model hyperparameter
            arc_seq = arc_seq.write(self.__start_id+2*i, index)
            
            # a layer block controller loss
            curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=index)
            log_prob += curr_log_prob
            
            # compute entropy of actions. but not update entropy to model. use only as reward.
            curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.nn.softmax(logits)))
            entropy += curr_ent
            
            # make index array integer using total sum
            prev_layers.append(anchors.read(tf.reduce_sum(index)))
            inputs = prev_layers[-1]
            
        return inputs, next_c, next_h, arc_seq, entropy, log_prob
        
    def __create_op_sampler(self, inputs, prev_c, prev_h, arc_seq, entropy, log_prob):
        # Softmax RNN Controller - for loop 2 : decide operation of nodes.
        for i in range(2):
            next_c, next_h = rnn_layer(inputs, prev_c, prev_h, self.w_recurrent)
            next_c, next_h = prev_c, prev_h
        
            #logits = tf.matmul(next_h[-1], self.w_soft)
            b_init = np.array([10.0, 10.0] + [0] * (self.num_branch - 2), dtype=np.float32)
            logits = tf.layers.dense(inputs=next_h[-1], units=self.num_branch, use_bias=True, name='Softmax_weighted', 
                                     bias_initializer=tf.constant_initializer(b_init) ,reuse=tf.AUTO_REUSE)
            
            # Softmax temperature (Softer Softmax)
            if self.temperature is not None : 
                logits /= self.temperature
            
            # Logits clipping
            if self.tanh_constant is not None : 
                logits = self.tanh_constant * tf.tanh(logits)
            
            # get best logit and select best layer index to use
            op_id = tf.multinomial(logits, 1)
            op_id = tf.to_int32(op_id)
            op_id = tf.reshape(op_id, [1])
            
            # write operation id
            arc_seq = arc_seq.write(self.__start_id+2*i+1, op_id)
            
            # log probabilty of actions; controller loss
            curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=op_id)
            log_prob += curr_log_prob
            
            # compute entropy of actions. but not update entropy to model. use only as reward.
            curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.nn.softmax(logits)))
            entropy += curr_ent
            
            # get current operation embedding vector as next rnn unit input.
            inputs = tf.nn.embedding_lookup(self.branch_vector, op_id)
            
        return inputs, next_c, next_h, arc_seq, entropy, log_prob
        
    def __create_trainer(self):
        
        # get model performances - Accuracy
        self.reward = tf.placeholder(tf.float32, name='controller_reward')
        
        # multiply controller entropy as reward. you can apply entropy when weight is given.
        if self.entropy_weight is not None : 
            self.reward += self.entropy_weight * self.sample_entropy
            
        # Controller rnn log prob - Controller rnn loss
        self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
        
        # Baseline of Reward (RL)
        self.baseline = tf.Variable(0., dtype=tf.float32, trainable=False)
        update_policy = (1 - self.bl_dec) * (self.baseline - self.reward)
        
        baseline_update = tf.assign_sub(self.baseline, update_policy)
        
        with tf.control_dependencies([baseline_update]):
            self.reward = tf.identity(self.reward)
            
        # final model loss
        self.loss = self.sample_log_prob * (self.reward - self.baseline)
        # get controller's weights only to train.
        vars_to_train = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        
        # Create optimizer with given learning rate update policy.
        self.train_op, self.lr, self.grad_norm, self.optimizer = set_optimizer(
            self.loss,
            vars_to_train,
            self.train_step,
            lr_init=self.lr_init,
            optimizer_type=self.optimizer_type,
            lr_options=self.lr_options,
            grad_options=self.grad_options,
        )
        
        # Skip rate?
        self.skip_rate = tf.constant(0., dtype=tf.float32)