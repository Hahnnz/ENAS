import tensorflow as tf

def rnn_layer(data, prev_c, prev_h, weights, unit_type='lstm'):
    '''
    args :
        - data  : 

    description : 


    '''
    unit_type = unit_type.lower()
    if unit_type == 'lstm':
        assert prev_c is not None, "'prev_c' is None. plz input prev_c"
    elif unit_type == 'gru':
        prev_c = [None for _ in range(prev_h)]
    else : raise ValueError("'lstm' and 'gru' are your options. plz check the unitype again")

    next_h = []
    next_c = []
    for layer_id, (c, h, w) in enumerate(zip(prev_c, prev_h, weights)): 
        data = data if layer_id == 0 else next_h[-1]
        curr_c, curr_h = recurrent(data, c, h, w, unit_type)
        next_c.append(curr_c)
        next_h.append(curr_h)
    if unit_type == 'lstm':
        return next_c, next_h
    elif unit_type == 'gru':
        return next_h

def recurrent(data, prev_c, prev_h, weight, utype='lstm'):
    if utype == 'lstm' : 
        ifog = tf.matmul(tf.concat([data, prev_h], axis=1), weight)
        i,f,o,g = tf.split(ifog, 4, axis=1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        next_c = i * g + f * prev_c
        next_h = o * tf.tanh(next_c)

        return next_c, next_h
    elif utype == 'gru':
        w_r, w_z, w_g = tf.split(rzg, 3, axis=1)
        _in = tf.concat([data, prev_h], axis=1)

        r = tf.sigmoid(tf.matmul(_in, w_r))
        z = tf.sigmoid(tf.matmul(_in, w_z))
        g = tf.tanh(tf.matmul(r*z, w_z))

        next_h = z*_in + (1-z)*g

        return next_h, prev_h

# Child model operator (Search Space)
    
def conv(data, ksize, filters, ssize, padding, use_bias, conv_mode='2d' ,conv_name=None):
    conv_mode = conv_mode.lower()
    if conv_mode == '2d':
        output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                                  strides=(ssize,ssize),
                                  padding=padding.upper(),
                                  name=conv_name,use_bias=use_bias,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
    elif conv_mode == '3d':
        output = tf.layers.conv3d(data, kernel_size=ksize, filters=filters,
                                  strides=(ssize,ssize,ssize),
                                  padding=padding.upper(),
                                  name=conv_name,use_bias=use_bias,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
    else :
        raise ValueError('Convoltion mode : ['+conv_mode+'] is not available. plz select "2d" or "3d".')
    return output

def dense(data, num_out, name=None, use_bias=True, trainable=True):
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

def max_pooling(data, ksize=3, ssize=2, mode='2d', padding='SAME', name=None):
    with tf.name_scope(name):
        if mode == '2d':
            return tf.nn.max_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding="SAME", name=name)
        elif mode == '3d':
            return tf.nn.max_pool3d(data, ksize=[1,ksize,ksize,ksize,1], strides=[1,ssize,ssize,ssize,1], padding="SAME", name=name)
        else :
            raise ValueError('Max Pooling mode : ['+conv_mode+'] is not available. plz select "2d" or "3d".')

def avg_pooling(data, ksize=3, ssize=2, mode='2d', padding='SAME', name=None):
    with tf.name_scope(name):
        if mode == '2d' : 
            return tf.nn.avg_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding=padding, name=name)
        elif mode == '3d' :
            return tf.nn.avg_pool3d(data, ksize=[1,ksize,ksize,ksize,1], strides=[1,ssize, ssize,ssize,1], padding=padding, name=name)