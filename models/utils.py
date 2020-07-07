import tensorflow as tf

def set_optimizer(loss, learning_rate, graph=None, optimizer_type='adam',):
    optimizer_type = optimizer_type.lower()
    if graph is None : graph = tf.get_default_graph()
    
    with graph.as_default():
        if optimizer_type=="adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer_type == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.0001)
        elif optimizer_type == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        elif optimizer_type == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=0.95,epsilon=1e-09)
        elif optimizer_type == "radam":
            optimizer = RAdamOptimizer(learning_rate=learning_rate)
        else : raise ValueError("{} optimizer isn't supported.".format(optimizer_type))
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        vars_to_train = tf.trainable_variables()
                
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_op)
                
    return train_op