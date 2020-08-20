import tensorflow as tf

from .layers import *
from .utils import *

# separable_convolution 2D & 3D
def separable_conv(tensor, k_size, out_fsize, s_size, padding, use_bias, conv_mode='2d', conv_name=None):
    num_in_chn = tensor.get_shape()[-1].value
    
    if conv_mode == '2d':
        tensor = tf.layers.separable_conv2d(tensor, kernel_size=k_size, filters=out_fsize,
                                            strides=s_size, padding=padding.upper(), use_bias=use_bias,
                                            depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                            pointwise_initializer=tf.contrib.layers.xavier_initializer())
    
    elif conv_mode == '3d':
        # Depthwise
        depthwise_result = []

        for idx in range(num_in_chn):
            in_tensor = tf.expand_dims(tensor[:,:,:,:,idx],-1)
            out_tensor = tf.layers.conv3d(in_tensor, kernel_size=k_size, filters=1,
                                          strides=s_size, padding=padding.upper(),
                                          use_bias=use_bias, reuse=tf.AUTO_REUSE,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
            depthwise_result.append(out_tensor)
        depthwise_result = tf.concat(depthwise_result,-1)

        # Pointwise
        tensor = tf.layers.conv3d(depthwise_result, kernel_size=1, filters=out_fsize,
                                  strides=1, padding=padding.upper(),
                                  use_bias=use_bias,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        
    return tensor
    
class Child:
    '''
    Classification Child network
    
    All Child Class function assume data format is 'NHWC'.
    '''
    def __init__(self,
                 input_shape,
                 num_classes,
                 lr_options,
                 grad_options,
                 out_fsize,
                 opt_algo='adam',
                 fixed_arc=None,
                 batch_size=None,
                 num_stem_out=2,
                 num_cells=5,
                 num_layers=6,
                 reuse=False,
                 phase='train',
                 model_name='Child'):
        
        phase = phase.lower()

        # check arguments
        assert type(batch_size) in [type(None),int], 'batch_size must be integer or None.'
        assert type(input_shape) in [tuple,list], 'input shape must be given as list or tuple.'
        assert phase in ['train','valid'], 'phase can be "train" or "valid" only.'

        if len(input_shape) == 3 :
            self.conv_mode = '2d'
        elif len(input_shape) == 4 :
            self.conv_mode = '3d'
        else :
            raise ValueError('input shape must have length of 3(w,h,chn) or 4(w,h,d,chn).')

        self.batch_size = batch_size
        self.optimizer_type = opt_algo
        self.__num_stem_out = num_stem_out # How many outputs that you want to create at Stage1.
        self.__input_shape = input_shape
        self.num_layers = num_layers # number of layers to search
        self.out_fsize = out_fsize
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.fixed_arc = fixed_arc
        self.model_name = model_name
        
        self.lr_options = lr_options
        self.grad_options = grad_options
        
        '''
        # Child model placeholders
        with tf.variable_scope(model_name, reuse=reuse):
            self.input_x = tf.placeholder(tf.float32, input_shape, name='input_x')
            self.targets = tf.placeholder(tf.int32, target_shape, name='target')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
            self.keep_prob = tf.placeholder(tf.float32, name='alive_ratio')
            self.lr_init = tf.placeholder(tf.float32, name='initial_learning_rate')
            
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        '''
    def initialize(self):
        with tf.variable_scope(self.model_name):
            
            # merge batch size with input shape. (N,W,H,C) or (N,W,H,D,C) if 3D
            shape_type = type(self.__input_shape)
            input_shape = shape_type([self.batch_size]) + self.__input_shape
            target_shape = shape_type([self.batch_size]) # Classification only for now
            
            self.input_x = tf.placeholder(tf.float32, input_shape, name='input_x')
            self.targets = tf.placeholder(tf.int32, target_shape, name='target')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
            self.keep_prob = tf.placeholder(tf.float32, name='alive_ratio')
            self.lr_init = tf.placeholder(tf.float32, name='initial_learning_rate')
            
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
            
            self.logits = self.__model()
            #one_hot_labels = tf.one_hot(self.target,depth=self.num_classes,axis=-1,dtype=tf.int32)
            #one_hot_labels = tf.squeeze(one_hot_labels)
            #log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_labels)
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            self.loss = tf.reduce_mean(log_probs)

            self.train_preds = tf.argmax(self.logits, axis=-1)
            self.train_preds = tf.to_int32(self.train_preds)
            self.train_acc = tf.equal(self.train_preds, self.targets)
            self.train_acc = tf.to_int32(self.train_acc)
            self.train_acc = tf.reduce_sum(self.train_acc)


            vars_to_train = [ var for var in tf.trainable_variables() if (
                var.name.startswith(self.model_name) and "aux_head" not in var.name)]

            self.train_op, self.lr, self.grad_norm, self.optimizer = set_optimizer(
                self.loss,
                vars_to_train,
                self.global_step,
                lr_init=self.lr_init,
                optimizer_type=self.optimizer_type,
                lr_options=self.lr_options,
                grad_options=self.grad_options,
            )
    
    def __model(self,reuse=False):
        # Stage.1 Stem Convolution - First two inputs
        with tf.variable_scope('stem_conv', reuse=reuse):
            tensor = conv(self.input_x,
                          ksize=3, # kernel_size
                          filters=self.out_fsize*3, # number of out-channel
                          ssize=1, # number of strides
                          padding='SAME',# padding
                          use_bias=False,
                          conv_mode=self.conv_mode) # 2D or 3D

            tensor = batch_norm(tensor, self.is_train)

        # copy first convolution layer output with number of first stem output.
        candidate_layers = [tensor for _ in range(self.__num_stem_out)]
        
        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2 * pool_distance + 1]

        # Stage 2. create Micro Search Space
        out_fsize = self.out_fsize
        for layer_id in range(self.num_layers+2):
            with tf.variable_scope(f"layer_{layer_id}"):
                # Common layer
                if layer_id not in self.pool_layers:
                    # select layer to use. controlling layer or fixed layer.
                    chosen_layer = self._control_layer if self.fixed_arc is None else self.fixed_layer
                    tensor = chosen_layer(layer_id,
                                          candidate_layers, # possible inputs
                                          self.normal_arc, #? 
                                          out_fsize, # number of output channel size
                                          )
                # Downsampling step
                else :
                    out_fsize *= 2
                    if self.fixed_arc is None :
                        tensor = self._factorized_reduction(tensor, out_fsize, 2)
                        candidate_layers = [candidate_layers[-1], tensor]

                        tensor = self._control_layer(layer_id,
                                                     candidate_layers,
                                                     self.reduce_arc,
                                                     out_fsize)
                    else : 
                        tensor = self.fixed_layer(candidate_layers, # possible inputs
                                                  layer_id,
                                                  self.normal_arc, #? 
                                                  out_fsize, # number of output channel size
                                                  self.is_train)
                print("Layer {0:>2d}: {1}".format(layer_id, tensor))
                candidate_layers = [candidate_layers[-1], tensor]

                ''' # Need Aux_head implimentation
                self.num_aux_vars = 0
                get_auxiliary = all([hasattr(self, 'use_aux_heads'), layer_id in self.aux_head_indices, is_training])

                if get_auxiliary :
                    with tf.variable_scope('aux_head'):
                        aux_logits = tf.nn.relu(tensor)
                        aux_logits = avg_pooling(aux_logits, ksize=5, ssize=3, mode=self.conv_mode, padding='VALID')

                        with tf.variable_scope('projection'):
                            aux_logits = conv(aux_logits, 1, 128, 1, 'SAME', False, self.conv_mode)
                            aux_logits = batch_norm(aux_logits, self.is_train)
                            aux_logits = tf.nn.relu(aux_logits)

                        with tf.variable_scope('avg_pool'):
                '''
        tensor = tf.nn.relu(tensor)
        tensor = global_avg_pooling(tensor, mode=self.conv_mode)
        
        '''
        do_dropout = all(self.is_train and self.keep_prob is not None and self.keep_prob < 1.0)
        if do_dropout :
            tensor = tf.nn.dropout(tensor, self.keep_prob)
        '''
        do_dropout = tf.stack([[tf.equal(self.is_train, tf.constant(True, tf.bool))], 
                               [tf.not_equal(tf.size(self.keep_prob), tf.constant(0, tf.int32))], 
                               [tf.less(self.keep_prob, tf.constant(0, tf.float32))]])
        do_dropout = tf.cast(do_dropout, tf.int32)
        dropout_cond = tf.constant([1,1,1,], tf.int32, shape=(1,3))
        
        do_dropout = tf.matmul(dropout_cond, do_dropout)
        do_dropout = tf.cast(tf.squeeze(do_dropout), tf.bool)
        
        def __dropout() : return tf.nn.dropout(tensor, self.keep_prob)
        def __indentity() : return tensor
        
        tensor = tf.cond(do_dropout,
                         true_fn=__dropout,
                         false_fn=__indentity)
        
        tensor = dense(tensor, self.num_classes, name='logits', use_bias=False)
        return tensor

    def _factorized_reduction(self, tensor, out_fsize, s_size):
        '''
        Description :
        Arguments : 
            - tensor :
            - out_fsize :
            - s_size :
        Outputs : 
        '''

        if s_size == 1 :
            with tf.variable_scope('path_conv'):
                tensor = conv(tensor, 1, out_fsize, s_size, 'SAME', False, self.conv_mode ,conv_name='conv')
                tensor = batch_norm(tensor, self.is_train)
                return tensor

        # Skip Path 1
        p1_tensor = avg_pooling(tensor, ksize=1, ssize=s_size, mode=self.conv_mode, padding='VALID')
        with tf. variable_scope('path1_conv'):
            p1_tensor = conv(p1_tensor, 1, out_fsize//2, 1, 'VALID', False, self.conv_mode ,conv_name='conv')

        # Skip Path 2
        if self.conv_mode == '2d':
            p2_tensor = tf.keras.layers.ZeroPadding2D((1,1))(tensor)[:,2:,2:,:]
        elif self.conv_mode == '3d':
            p2_tensor = tf.keras.layers.ZeroPadding3D((1,1,1))(tensor)[:,2:,2:,2:,:]
        '''
        _pad = [[0, 0], [0, 1], [0, 1], [0, 0]]
        if self.conv_mode == '2d':
            p2_tensor = tf.pad(tensor, _pad)[:, 1:, 1:, :]
        elif self.conv_mode == '3d':
            padding_3d = []
            with tf.variable_scope('3d_padding')
            for d in range(tens):
            
            
            
            p2_tensor = tf.pad(tensor, _pad)[:, 1:, 1:, 1:, :]'''
        p2_tensor = avg_pooling(p2_tensor, ksize=1, ssize=s_size, mode=self.conv_mode, padding='VALID')
        with tf.variable_scope("path2_conv"):
            p2_tensor = conv(p2_tensor, 1, out_fsize//2, 1, 'VALID', False, self.conv_mode, conv_name='conv')

        tensor = tf.concat([p1_tensor,p2_tensor],-1)
        tensor = batch_norm(tensor, self.is_train)
        
        return tensor

    def _calibration(self, layers, out_fsize, target_layer_idx=-1):
        '''
        Description :
        Arguments : 
        Outputs : 
        '''
        src_layer, target_layer = layers
        src_shape = src_layer.shape.as_list()[1:]
        target_shape = target_layer.shape.as_list()[1:]

        with tf.variable_scope('calibraion'):
            # Source Layer Shape Calibration
            if src_shape[:-1] != target_shape[:-1]:
                bigger_than_2 = [s1//s2 for s1, s2 in zip(src_shape[:-1],target_shape[:-1])]
                bigger_than_2 = [d for d in bigger_than_2 if d != 2]
                assert not bigger_than_2, 'error. need to add msg about 2time bigger'
                with tf.variable_scope('src_shape_calibration'):
                    src_layer = tf.nn.relu(src_layer)
                    src_layer = self._factorized_reduction(src_layer, out_fsize, 2)

            # Source Layer Filter size Calibration
            if out_fsize != src_shape[-1]:
                src_layer = tf.nn.relu(src_layer)
                src_layer = conv(src_layer, 1, out_fsize, 1, 'SAME', False, self.conv_mode)
                src_layer = batch_norm(src_layer, self.is_train)

            # Target Layer Filter size Calibration
            if out_fsize != target_shape[-1]:
                target_layer = tf.nn.relu(target_layer)
                target_layer = conv(target_layer, 1, out_fsize, 1, 'SAME', False, self.conv_mode)
                target_layer = batch_norm(target_layer, self.is_train)

            return [src_layer, target_layer]

    def _create_control_cell(self, tensor, curr_cell, prev_cell, op_id, out_fsize, num_conv_stack=2):
        '''
        Description :
        Arguments : 
        Outputs : 
        '''

        # 모르겠음 역할이 뭐지?
        num_possible_inputs = curr_cell +1

        def __calibrate_fsize(tensor, fsize, name=None):
            '''calibrate tensor out-filter size'''
            if name is None : name = 'conv_calib'

            with tf.variable_scope(name):
                tensor = tf.nn.relu(tensor)
                tensor = conv(tensor, 1, fsize, 1, 'SAME', False, self.conv_mode)
                tensor = batch_norm(tensor, self.is_train)
            return tensor

        out_cell_op = []

        # Operation 1 & 2 : Convolution 3x3 , 5x5
        conv_ksize_list = [3, 5]
        __conv_name = 'conv_{0}x{0}'
        for k_size in conv_ksize_list:
            with tf.variable_scope(__conv_name.format(k_size)): 
                for conv_id in range(num_conv_stack):
                    conv_out = tf.nn.relu(tensor)
                    conv_out = separable_conv(conv_out, k_size, out_fsize, 1, 'SAME', False, self.conv_mode)
                    conv_out = batch_norm(conv_out, self.is_train)
                    out_cell_op.append(conv_out)

        # Operation 3 : Average Pooling
        with tf.variable_scope("Avg_pooling"): 
            avg_pool = avg_pooling(tensor, ksize=3, ssize=1, mode=self.conv_mode, padding='SAME')
            if avg_pool.get_shape()[-1].value != out_fsize:
                avg_pool = __calibrate_fsize(avg_pool, out_fsize)
            out_cell_op.append(avg_pool)

        # Operation 4 : Max Pooling
        with tf.variable_scope("MAX_pooling"): 
            max_pool = max_pooling(tensor, ksize=3, ssize=1, mode=self.conv_mode, padding='SAME')
            if max_pool.get_shape()[-1].value != out_fsize:
                max_pool = __calibrate_fsize(max_pool, out_fsize)
            out_cell_op.append(max_pool)

        # Operation 5 : Identity
        if tensor.get_shape()[-1].value != out_fsize:
            tensor = __calibrate_fsize(tensor, out_fsize,'x_calib')
        out_cell_op.append(tensor)

        out_cell_op = tf.stack(out_cell_op, axis=0)
        out_cell_op = out_cell_op[op_id]

        return out_cell_op

    def _control_conv(self, tensor, curr_cell, prev_cell, k_size, out_fsize, stack_conv=2):
        '''
        Description :
        Arguments : 
        Outputs : 
        '''
        with tf.variable_scope('conv_{0}x{0}'.format(k_size)):
            num_possible_inputs = curr_cell +2

            for conv_id in range(stack_conv):
                with tf.variable_scope(f'stack_{conv_id}'):
                    tensor = tf.nn.relu(tensor)
                    tensor = separable_conv(tensor, k_size, out_fsize, 1, 'SAME', False, self.conv_mode)
                    tensor = batch_norm(tensor, self.is_train)

            return tensor

    def _control_layer(self, layer_id, prev_layers, arc, out_fsize):
        '''
        Description :
        Arguments : 
        Outputs : 
        '''

        layers = [prev_layers[0],prev_layers[1]]
        layers = self._calibration(layers, out_fsize)
        
        used = []
        result = []
        
        for c_id in range(self.num_cells):
            prev_layers = tf.stack(layers, axis=0)
            with tf.variable_scope(f'cell_{c_id}'):
                # compute input with given operation
                for idx in range(2):
                    with tf.variable_scope(f'input_{idx}'):
                        inp_id = arc[ (4*c_id) + (idx*2) ]
                        inp_op = arc[ (4*c_id) + (idx*2) +1 ]
                        
                        tensor = prev_layers[inp_id]
                        tensor = self._create_control_cell(tensor, c_id, inp_id, inp_op, out_fsize)
                        result.append(tensor)

                        tensor_used = tf.one_hot(inp_id, depth = self.num_cells+2, dtype=tf.int32)
                        used.append(tensor_used)

                layers.append(sum(result))

        used = tf.add_n(used)
        indices = tf.where(tf.equal(used, 0))
        indices = tf.to_int32(indices)
        indices = tf.reshape(indices, [-1])
        num_outs = tf.size(indices)

        out = tf.stack(layers, axis=0)
        out = tf.gather(out, indices, axis=0)
        
        inp = prev_layers[0]
        orig_shape = inp.get_shape().as_list()
        out_shape = out.get_shape().as_list()
        
        
        N = tf.shape(inp)[0]
        H = tf.shape(inp)[1]
        W = tf.shape(inp)[2]
        C = tf.shape(inp)[3]        
        
        if self.conv_mode == '2d':
            shape_reorder = [1,2,3,0,4]
        else :
            shape_reorder = [1,2,3,4,0,5] 
        
        out = tf.transpose(out, shape_reorder)
        out = tf.reshape(out, [N,H,W,num_outs*out_fsize])
        
        with tf.variable_scope('final_conv'):
            initializer = tf.contrib.keras.initializers.he_normal()
            w = tf.get_variable('weight', [self.num_cells + 2, out_fsize * out_fsize], initializer=initializer)
            w = tf.gather(w, indices, axis=0)
            w = tf.reshape(w, [1, 1, num_outs * out_fsize, out_fsize])
            out = tf.nn.relu(out)
            out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME")
            out = batch_norm(out, self.is_train)
            '''
            out = tf.nn.relu(out)
            out = conv(out, 1, out_fsize, 1, 'SAME', False, self.conv_mode)
            out = batch_norm(out, self.is_train)
            '''
        out = tf.reshape(out, tf.shape(prev_layers[0]))
        return out
    
    def connect_controller(self, controller_model):
        if self.fixed_arc is None :
            self.normal_arc, self.reduce_arc = controller_model.sample_arc
        else :
            fixed_arc = np.array([int(x) for x in self.fixed_arc.split(' ') if x])
            self.normal_arc = fixed_arc[:4 * self.num_cells]
            self.reduce_arc = fixed_arc[4 * self.num_cells:]

###########################
# Fixed Layer Zone. [WIP] #
###########################

'''
    def _common_conv(self, tensor, k_size, num_out_chn, s_size, num_stack=2):
        for idx in range(stack_convs):
            s_size = s_size if conv_id == 0 else 1
            with tf.variable_scope(f"sep_conv_{idx}"):
                tensor = tf.nn.relu(tensor)
                tensor = separable_conv(tensor, k_size, num_out_chn, s_size, padding='SAME', use_bias=False, self.conv_mode)
                tensor = batch_norm(tensor, self.is_train)
        return tensor

    def _concat_layers(self, layers, is_used, num_out_chn, normal_or_reduction_cell="normal"):

        with tf.variable_scope('concatenated'):
            layers2concat = []
            for idx, (tensor, used) in enumerate(zip(layers, is_used)):
                if not used :
                    num_in_chn = get_size(tensor, 'c')
                    if num_out_chn != num_in_chn:
                        assert hw == out_hw * 2, (f"output shape should be bigger double times than input size.")
                        with tf.varialbe_scope(f'HW_calibration_{idx}'):
                            tensor = self._factorized_reduction(tensor, num_out_chn, 2)
                    layers2concat.append(tensor)

            tensor = tf.concat(layers2concat, -1)

        return tensor

    def _common_layer(self, layer_id, prev_layers, arc, num_out_chn, s_size, normal_or_reduction_cell="normal"):

        assert len(prev_layes) == 2, 'previous layers must have 2 layers'
        prev_layers = self._calibration(prev_layers, num_out_chn)

        with tf. variable_scope('layer_base'):
            tensor = prev_layers[1]
            num_in_chn = get_size(tensor, 'c')

            tensor = tf.nn.relu(tensor)
            tensor = conv(tensor, 1, num_out_chn, 1, 'SAME', '3d', False, self.conv_mode)
            tensor = batch_norm(tensor, self.is_train)

            prev_layers[1] = tensor

        used = np.zeros([self.num_cells + 2], dtype=np.int32)
        kernel_sizes = [3, 5]
        for cell_id in range(self.num_cells):
            with tf.variable_scope(f'cell_{cell_id}'):
                cell_start = 4 * cell_id
                seq_id = arc[cell_start]
                used[seq_id] += 1

                seq_op = arc[cell_start + 1]
                tensor = layers[seq_op]
                # convolution cell operation id
                s_size = s_size if seq_id in [0, 1] else 1

                # Operations
                with tf.variable_scope('x_conv'):
                    if seq_op in [0,1]:
                        k_size = kernel_sizes
                        tensor = self._common_conv(tensor, k_size, num_out_chn, s_size)
                    elif seq_op == 2:
                        tensor = avg_pooling(tensor, ksize=3, ssize=ssize, mode=self.conv_mode, padding='SAME')
                    elif seq_op == 3:
                        tensor = max_pooling(tensor, ksize=3, ssize=ssize, mode=self.conv_mode, padding='SAME')





        raise NotImplementedError("WIP")
        '''