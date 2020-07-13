import tensorflow as tf

def recurrent(num_hidden, 
              cell_type='lstm', # LSTM or GRU
              do_bidirect=False, 
              # Activate Dropout
              enable_drop=False, 
              keep_prob=1, # Alive-ratio
              # Activate and set Attention
              enable_attn=False, 
              attn_length=0, 
              # Activate Residual
              enable_residual=False, 
              # Name
              name_postfix=''):
    
    cell_type = cell_type.lower()
    
    # get Recurrent unit with the type you want
    if cell_type == 'lstm':
        rnn_cell = tf.nn.rnn_cell.LSTMCell
    elif cell_type == 'gru':
        rnn_cell = tf.nn.rnn_cell.GRUCell
    else :
        raise ValueError(f'{cell_type} is not support! plz choose one of following cell type : "lstm", "gru".')
        
    # Name
    layer_name = f'{cell_type}_layer' + name_postfix
    if do_bidirect : 'bi'+layer_name
    
    # create recurrent unit.
    with tf.variable_scope(layer_name) as scope:
        forward_cell = rnn_cell(num_hidden, name='forward_cell')
        
        # Apply additional ops.
        if enable_drop :
            forward_cell = tf.nn.rnn_cell.DropoutWrapper(forward_cell, keep_prob)
        if enable_residual : 
            forward_cell = tf.nn.rnn_cell.ResidualWrapper(forward_cell)
        if enable_attn : 
            forward_cell = tf.contrib.rnn.AttentionCellWrapper(forward_cell, attn_length)
            
        # For bidirectional unit.
        if do_bidirect : 
            backward_cell = rnn_cell(num_hidden, name='backward_cell')
            backward_cell = dropout(backward_cell, keep_prob)
            if enable_residual : 
                backward_cell = residal(backward_cell)
            if enable_attn : 
                backward_cell = attn(backward_cell, attn_length)
            return forward_cell, backward_cell
        else : 
            return forward_cell
    
'''    
def creaste_weights(name, shape, initializer=None, trainable=True, seed=None):
    raise NotImplementedError("Abstract method.")
    
def creaste_bias(name, shape, initializer=None):
    raise NotImplementedError("Abstract method.")
'''    
def dense(data, num_out, name=None, bn=True, use_bias=True, trainable=True):
    with tf.variable_scope(name) as scope:
        output = tf.layers.dense(inputs=data, use_bias=use_bias, units=num_out, trainable=trainable)
    return output

def batch_norm(data, is_train, trainable=True, name=None, data_format='channels_last',
               USE_FUSED_BN = True, BN_EPSILON = 1e-3, BN_MOMENTUM = 0.99):
    bn_axis = -1 if data_format == 'channels_last' else 1
    return tf.layers.batch_normalization(data, training=is_train, name=name, momentum=BN_MOMENTUM, axis=bn_axis,
                                         trainable=trainable, epsilon=BN_EPSILON, reuse=None, fused=USE_FUSED_BN)
    
def global_avg_pooling(data, mode='2d', name=None):
    with tf.name_scope(name):
        if mode == '2d' : 
            global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        elif mode == '3d' : 
            global_avg_pool = tf.keras.layers.GlobalAveragePooling3D()
        else :
            raise ValueError("'mode' must be '2d' or '3d'.")
        return global_avg_pool(data)
    
def dropout(data, keep_prob, name=None):
    with tf.name_scope(name):
        return tf.nn.dropout(data, keep_prob, name=name)
    
def max_pooling(data, ksize=3, ssize=2, mode='2d', name=None):
    with tf.name_scope(name):
        if mode == '2d':
            return tf.nn.max_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding="SAME", name=name)
        elif mode == '3d':
            return tf.nn.max_pool3d(data, ksize=[1,ksize,ksize,ksize,1], strides=[1,ssize,ssize,ssize,1], padding="SAME", name=name)
        else :
            raise ValueError('Max Pooling mode : ['+conv_mode+'] is not available. plz select "2d" or "3d".')

def avg_pooling(data, ksize=3, ssize=2, mode='2d', name=None):
    with tf.name_scope(name):
        if mode == '2d' : 
            return tf.nn.avg_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding="VALID", name=name)
        elif mode == '3d' :
            return tf.nn.avg_pool3d(data, ksize=[1,ksize,ksize,ksize,1], strides=[1,ssize, ssize,ssize,1], padding="VALID", name=name)