import os
import numpy as np
import tensorflow as tf
from .layers.enas import *
from .layers.ops import *
from .utils import set_optimizer

class Controller():
    def __init__(self, batch_size=None, 
                 utype='lstm', # recurrent unit type
                 otype='adam', # optimizer type
                 num_layers=2, # Number of Recurrent unit layer
                 unit_dim=32, # number of channel of recurrent unit
                 keep_prob=1., # dropout ratio
                 tanh_constant=None,
                 entropy_weight=None,
                 op_tanh_reduce = 1., 
                 seed=None,
                 name='Controller'):
        
        self.batch_size = batch_size
        self.optimizer_type = otype # optimizer type
        self.unit_type = utype # recurrent unit type
        self.num_layers = num_layers # Number of Recurrent unit layer
        self.unit_dim = unit_dim # number of channel of recurrent unit
        self.num_branch = num_branch # number of hyperparameter you want
        self.keep_prob = keep_prob # dropout ratio
        self.entropy_weight = entropy_weight # entropy weight of reward entropy
        self.op_tanh_reduce = op_tanh_reduce # 
        self.name = name
        
        # Global controller train step
        self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")
        # initialize Controller Recurrent unit weights
        __init_controller()
        
        
        
        raise NotImplementedError("WIP")
        
    def __init_controller(self):
        with tf.variable_scope(self.name):
            
            # create recurrent unit layer with initializing 
            self.w_recurrent = []
            for l_id in range(self.num_layers):
                with variable_scope(f'{self.unit_type}_layer_{l_id}'):
                    __num_w_mat = 4 if self.unit_type == 'lstm' else 3
                    w = tf.get_variable('weights', [2*self.unit_dims , __num_w_mat*self.unit_dims])
                    self.w_recurrent.append(w)
            
            # create generator embeddings for each branch
            self.embedding = tf.get_variable('generator_embeddings', [1, self.unit_dim])
            with tf.variable_scope('embedding'):
                self.w_emb = tf.get_variable('weights', [self.num_branch, self.unit_dim])
            
            # Softmax
            with tf.variable_scope('softmax'):
                self.w_sotf  = tf.get_variable('weights', [self.unit_dim, self.num_branches])
                
                b_init = np.array([10.0, 10.0] + [0] * (self.num_branches - 2), dtype=np.float32)
                self.b_soft = tf.get_variable('bias', [1, self.num_branches], initializer=tf.constant_initializer(b_init))
                
                b_soft_no_learn = np.array([0.25, 0.25] + [-0.25] * (self.num_branches - 2), dtype=np.float32)
                b_soft_no_learn = np.reshape(b_soft_no_learn, [1, self.num_branches])
                self.b_soft_no_learn = tf.constant(b_soft_no_learn, dtype=tf.float32)
                
            with tf.variable_scope('attention'):
                self.w_attn_1 = tf.get_variable("weight_1", [self.lstm_size, self.lstm_size])
                self.w_attn_2 = tf.get_variable("weight_2", [self.lstm_size, self.lstm_size])
                self.v_attn = tf.get_variable("value", [self.lstm_size, 1])
    
    def __create_sampler(self, prev_c=None, prev_h=None, use_bias=False):
        # create anchor
        anchors = tf.TensorArray(tf.float32, size=self.num_cells + 2, clear_after_read=False)
        anchors_w1 = tf.TensorArray(tf.float32, size=self.num_cells + 2, clear_after_read=False)
        # model decision sequence
        arc_seq = tf.TensorArray(tf.int32, size=self.num_cells * 4)
        
        if prev_c is None :
            assert prev_h is None, "prev_c and prev_h must both be None or given"
            
            prev_c = [tf.zeros([1,self.unit_dim], tf.float32) for _ in range(self.num_layers)]
            prev_h = [tf.zeros([1,self.unit_dim], tf.float32) for _ in range(self.num_layers)]
        
        inputs = self.g_emb
        for layers_id in range(2):
            next_c, next_h = rnn_layer(inputs, prev_c, prev_h, self.w_recurrent)
            prev_c, prev_h = next_c, next_h
            
            anchors = anchors.write(layer_id, tf.zeros_like(next_h[-1]))
            anchors_w1 = anchors_w1.write(layer_id, tf.matmul(next_h[-1], self.w_attn_1))
        
    def __create_layer_controller(self, layer_id, data, prev_c, prev_h, anchor, anchor_w1, arc_seq, entorpy, log_prob):
        def condition(layer_id, *args):
            return tf.less(layer_id, self.num_cells + 2)
        indices = tf.range(0,layer_id, dtype=tf.int32)
        start_id = 4*(layer_id-2)
        
        prev_layers = []
        
        # Querying RNN Controller - for loop 1
        for i in range(2):
            # stack 2 layers for given layer ID.
            # this creates rnn layer that control a specific model layer only
            next_c, next_h = rnn_layer(data, prev_c, prev_h, self.w_recurrent)
            prev_c, prev_h = next_c, next_h
            
            # Query??? [IDK]
            query = anchors_w1.gather(indices)
            query = tf.reshape(query, [layer_id, self.unit_dim])
            query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
            query = tf.matmul(query, self.v_attn) # WHY Attention? [IDK]
            logits = tf.reshape(query, [1, layer_id])
            
            # 역할 모르겠음.
            if self.temperature is not None : 
                logits /= self.temperature
            
            # hypublic tangent????
            if self.tanh_constant is not None : 
                logits = self.tanh_constant * tf.tanh(logits)
                
            # get best logit and select best parameter to use
            index = tf.multinomial(logits, 1)
            index tf.to_int32(index)
            index = tf.reshape(index, [1])
            
            # write selected model hyperparameter
            arc_seq = arc_seg.write(start_id+2 * i, index)
            
            # a layer block controller loss
            curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=index)
            log_prob += curr_log_prob
            
            # a layer block controller entropy
            # freeze entropy
            curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.nn.softmax(logits)))
            entropy += curr_ent
            
            # 잘모르겠다. 다음에 찾아서 주석 달아야겠어
            prev_layers.append(anchors.read(tf.reduce_sum(index)))
            inputs = prev_layers[-1]
        
        # 이건 뭘까.. Operation 1, 2 를 정의하는 부분같긴한데.. 정확히 무엇을 정의하는지 모르겠음.
        # Softmax RNN Controller - for loop 1
        for i in range(2):
            next_c, next_h = rnn_layer(data, prev_c, prev_h, self.w_recurrent)
            next_c, next_h = prev_c, prev_h
        
            logits = tf.matmul(next_h[-1], self.w_soft)
            
            # Power Reducing 같은데 좀 더 보자.
            if self.temperature is not None : 
                logits /= self.temperature
            
            # hypublic tangent with Power Reducing
            if self.tanh_constant is not None : 
                logits = self.tanh_constant * tf.tanh(logits)
            
            op_id = tf.multinomial(logits, 1)
            op_id = tf.to_int32(op_id)
            op_id = tf.reshape(op_id, [1])
            
            arc_seq = arc_seq.write(start_id + 2 * i + 1, op_id)
            
            curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=op_id)
            log_prob += curr_log_prob
            
            curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.nn.softmax(logits)))
            entropy += curr_ent
            
            inputs = tf.nn.embedding_lookup(self.w_emb, op_id)
        
        loop_vars = [
            tf.constant(2, dtype=tf.int32, name='layer_id'),
            inputs, prev_c, prev_h,
            anchors, anchors_w1, arc_seq,
            tf.constant([0.,], dtype=tf.float32, name='entropy'),
            tf.constant([0.,], dtype=tf.float32, name='log_prob'),
        ]
        
        # Controller rnn cells until layer id reach the number of cells.
        loop_outputs = tf.while_loop(condition, __create_layer_controller, loop_vars, parallel_iterations=1)
        
        arc_srq = arc_srq[-3].stack()
        arc_srq = tf.reshape(arc_srq, [-1])
        entropy = tf.reduce_sum(loop_outputs[-2])
        log_prob = tf.reduce_sum(loop_outputs[-1])
        
        last_c = loop_outputs[-7]
        last_h = loop_outputs[-6]
        
        return arc_seq, entropy, log_prob, last_c, last_b
        
    def __create_trainer(self, child_model):
        child_model.build_valid_rl()
        
        # get model performances - Accuracy
        self.valid_acc = (tf.to_float(child_model.valid_shuffle_acc) /
                          tf.to_float(child_model.batch_size))
        self.reward = self.valid_acc
        
        # multiply controller entropy as reward. you can apply entropy when weight is given.
        if self.entropy_weight is not None : 
            self.reward += self.entropy_weight * self.sample_entropy
            
        # Controller rnn log prob - Controller rnn loss
        self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
        
        # Baseline of Reward (RL)
        self.baseline = tf.Variable(0., dtype=tf.float32, trainable=False)
        update_policy = (1 - self.bl_dec) * (self.baseline - self.reward)
        
        baseline_update = tf.assign_sub(self.baseline, update_policy)
        
        with tf.contorl_dependencies([baseline_update]):
            self.reward = tf.identity(self.reward)
            
        # final model loss
        self.loss = self.sample_log_prob * (self.reward - self.baseline)
        vars_to_train = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        
        
        self.train_op, self.lr, self.grad_norm, self.optimizer = set_optimizer(
            self.loss,
            vars_to_train,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            optim_algo=self.optim_algo,
            sync_replicas=self.sync_replicas,
            num_aggregate=self.num_aggregate,
            num_replicas=self.num_replicas
        )
        
        # Skip rate?
        self.skip_rate = tf.constant(0., dtype=tf.float32)