import tensorflow as tf
import numpy as np
from copy import deepcopy

class config_options:
    # Lerarning rate update policy of optimizer
    def lr(self,
           warmup_val=None,
           warmup_steps=1e3,
           decay_start=0,
           decay_every=1e4,
           decay_rate=1e-1,
           decay_min=None,
           lr_cosine=False,
           max_lr=None,
           min_lr=None,
           T_0=None,
           T_mul=None,):
        
        config = deepcopy(self)
        
        config.lr_warmup_val=warmup_val
        config.lr_warmup_steps=warmup_steps
        config.lr_dec_start=decay_start
        config.lr_dec_every=decay_every
        config.lr_dec_rate=decay_rate
        config.lr_dec_min=decay_min
        config.lr_cosine=lr_cosine
        config.lr_max=max_lr
        config.lr_min=min_lr
        config.lr_T_0=T_0
        config.lr_T_mul=T_mul
        return config
        
    # gradient update policy of optimizer

    def gradient(self,
                 l2_reg=1e-4,
                 clip_mode=None,
                 grad_threshold=None, 
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 get_grad_norms=False,
                 moving_average=None):
        
        config = deepcopy(self)
        
        config.l2_reg=l2_reg
        config.clip_mode=clip_mode
        config.grad_threshold=grad_threshold
        config.sync_replicas=sync_replicas
        config.num_aggregate=num_aggregate
        config.num_replicas=num_replicas
        config.get_grad_norms=get_grad_norms
        config.moving_average=moving_average
        return config
        
def set_optimizer(
    loss_op,
    vars2train,
    train_step,
    # clip gradient
    clip_mode=None,
    grad_threshold=None, 
    # L2 regularization
    l2_reg=1e-4,
    # Learning rate Policy
    lr_init=1e-1, # initial lerarning rate
    lr_warmup_val=None,
    lr_warmup_steps=1e3,
    lr_dec_start=0,
    lr_dec_every=1e4,
    lr_dec_rate=1e-1,
    lr_dec_min=None,
    lr_cosine=False,
    lr_max=None,
    lr_min=None,
    lr_T_0=None,
    lr_T_mul=None,
    # take learning rate above options without giving them except 'lr_init'
    lr_options=None,
    num_train_batches=None,
    optimizer_type="adam",
    # ????
    sync_replicas=False,
    num_aggregate=None,
    num_replicas=None,
    get_grad_norms=False,
    moving_average=None,
    grad_options=None):
    
    # take learning Rate options from configuration class
    if lr_options is not None :
        lr_warmup_val = lr_options.lr_warmup_val
        lr_warmup_steps = lr_options.lr_warmup_steps
        lr_dec_start = lr_options.lr_dec_start
        lr_dec_every = lr_options.lr_dec_every
        lr_dec_rate = lr_options.lr_dec_rate
        lr_dec_min = lr_options.lr_dec_min
        lr_cosine = lr_options.lr_cosine
        lr_max = lr_options.lr_max
        lr_min = lr_options.lr_min
        lr_T_0 = lr_options.lr_T_0
        lr_T_mul = lr_options.lr_T_mul 
        
    # take Optimization policy options from configuration class
    if grad_options is not None :
        l2_reg = grad_options.l2_reg
        clip_mode = grad_options.clip_mode
        grad_threshold = grad_options.grad_threshold
        sync_replicas = grad_options.sync_replicas
        num_aggregate = grad_options.num_aggregate
        num_replicas = grad_options.num_replicas
        get_grad_norms = grad_options.get_grad_norms
        moving_average = grad_options.moving_average
    
    # l2 regularization
    if l2_reg > 0 :
        l2_losses = [tf.reduce_sum(var**2) for var in vars2train]
        l2_losses = tf.add_n(l2_losses)
        loss_op += l2_reg * l2_losses
        
    # Compute gradients for given trainable variables
    gradients = tf.gradients(loss_op, vars2train)
    grad_norm = tf.global_norm(gradients)
    
    grad_norms = {}
    for var, grad in zip(vars2train, gradients):
        if var is None or grad is None : continue
        if isinstance(grad, tf.IndexedSlices):
            grad_norms[var.name] = tf.sqrt(tf.reduce_sum(grad.values ** 2))
        else : 
            grad_norms[var.name] = tf.sqrt(tf.reduce_sum(grad ** 2))
            
    # Clip gradient at given threshold
    if clip_mode is not None :
        assert grad_threshold is not None, 'Need "grad_threshold" to clip gradients'
        if clip_mode == 'global':
            gradients, _ = tf.tf.clip_by_global_norm(gradients, grad_threshold)
        elif clip_mode == 'norm':
            clipped = []
            for grad in gradients:
                if isinstance(gradients, tf.IndexedSlices):
                    c_grad = tf.clip_by_norm(grad.values, grad_threshold)
                    c_grad = tf.IndexedSlices(grad.indices, c_grad)
                else :
                    c_g = tf.clip_by_norm(grad, grad_threshold)
                    
                clipped.append(grad)
            gradients = clipped
    
    # Learning Rates Update Policy
    '''
    Bello, Irwan, Pham, Hieu, Le, Quoc V., Norouzi, Moham- mad, and Bengio, Samy. Neural combinatorial optimization with reinforcement learning. In ICLR Workshop, 2017a.
    https://arxiv.org/pdf/1611.09940.pdf
    
    Loshchilov, Ilya and Hutter, Frank. Sgdr: Stochastic gradient descent with warm restarts. In ICLR, 2017
    https://arxiv.org/pdf/1608.03983.pdf
    '''
    if lr_cosine :
        # check necessary arguments
        assert lr_max is not None, "Need lr_max to use lr_cosine"
        assert lr_min is not None, "Need lr_min to use lr_cosine"
        assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
        assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"
        assert num_train_batches is not None, ("Need num_train_batches to use 'lr_cosine'")
        
        curr_epoch = train_step // num_train_batches
        last_reset = tf.Variable(0, dtype=tf.int32, trainable=False, name='last_reset')
        
        T_i = tf.Variable(lr_T_0, dtype=tf.int32, trainable=False, name='T_i')
        T_curr = curr_epoch - last_reset
        
        def __do_update():
            update_last_reset = tf.assign(last_reset, curr_epoch, use_locking=True)
            update_T_i = tf.assign(T_i, T_i * lr_T_mul, use_locking=True)
            with tf.control_dependencies([update_last_reset, update_T_i]):
                rate = tf.to_float(T_curr) / tf.to_float(T_i) * np.pi
                lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
            return lr
                
        def __no_update():
            rate = tf.to_float(T_curr) / tf.to_float(T_i) * np.pi
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
            return lr
            
        learning_rate = tf.cond(tf.greater_equal(T_curr, T_i), __do_update, __no_update)
        
    else:
        learning_rate = tf.train.exponential_decay(
            lr_init, tf.maximum(train_step - lr_dec_start, 0), lr_dec_every,
            lr_dec_rate, staircase=True)
        
        if lr_dec_min is not None :
            learning_rate = tf.maximum(learning_rate, lr_dec_min)
            
    if lr_warmup_val is not None:
        learning_rate = tf.cond(tf.less(train_step, lr_warmup_steps),
                                lambda: lr_warmup_val, lambda: learning_rate)
        
    # set Optimizer Algorithm
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
        
        
    if sync_replicas:
        assert num_aggregate is not None, "Need num_aggregate to sync."
        assert num_replicas is not None, "Need num_replicas to sync."
        
        optimizer = tf.train.SyncReplicasOptimizer(optimizer, 
                                                   replicas_to_aggregate = num_aggregate, 
                                                   total_num_replicas = num_replicas,
                                                   use_locking=True)
    if moving_average is not None:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer, average_decay=moving_average)
        
    train_op = optimizer.apply_gradients(zip(gradients, vars2train), global_step=train_step)
    
    if get_grad_norms : 
        return train_op, learning_rate, grad_norm, optimizer, grad_norms
    else : 
        return train_op, learning_rate, grad_norm, optimizer