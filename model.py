# encoding: utf-8
import tensorflow as tf
import re
import numpy as np
import globals as g_


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', g_.BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', g_.INIT_LEARNING_RATE,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_float('clip_gradients', g_.GRADIENT_CLIPPING_THRESHOLD,
                            """Gradient clipping threshold.""")
tf.app.flags.DEFINE_float('likelihood_threshold', 0.0,
                            """Likelihood suppression threshold.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
WEIGHT_DECAY_FACTOR = 0.004 / 5. # 3500 -> 2.8

TOWER_NAME = 'tower'
DEFAULT_PADDING = 'SAME'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           initializer=tf.contrib.layers.xavier_initializer())
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv(name, in_ ,ksize, strides=[1,1,1,1], padding=DEFAULT_PADDING, group=1, reuse=False):
    
    n_kern = ksize[3]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides, padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            conv = convolve(in_, kernel)
        else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            input_groups = tf.split(in_, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in list(zip(input_groups, kernel_groups))]
            # Concatenate the groups
            conv = tf.concat(output_groups, 3)

        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name=scope.name)
        _activation_summary(conv)

    print(name, conv.get_shape().as_list())
    return conv

def _lrn(name, in_, bias=2, alpha=0.0001, beta=0.75, depth=5):
    lrn = tf.nn.lrn(in_, depth_radius=depth,
                    bias=bias, alpha=alpha, beta=beta)
    print(name, lrn.get_shape().as_list())
    return lrn

def _maxpool(name, in_, ksize, strides=[1, 1, 1, 1], padding=DEFAULT_PADDING):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print(name, pool.get_shape().as_list())
    return pool

def _avgpool(name, in_, ksize, strides=[1, 1, 1, 1], padding=DEFAULT_PADDING):
    pool = tf.nn.avg_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)
    
    print(name, pool.get_shape().as_list())
    return pool

def _fc(name, in_, outsize, dropout=1.0, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        
        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)

        _activation_summary(fc)

    

    print(name, fc.get_shape().as_list())
    return fc
    
def _softmax_likelihood(name, in_, eps=1e-5, rho=1e-2, threshold=0.0):
    prob = tf.nn.softmax(in_)
    logy = tf.nn.relu(-tf.log(tf.add(prob, eps)))
    likelihood = tf.reciprocal(tf.add(tf.reduce_sum(tf.multiply(prob, logy), -1, keepdims=True), rho))
    
    if threshold > 0.0:
        likelihood = tf.add(likelihood, -threshold)
        likelihood = tf.nn.relu(likelihood)

    print(name, likelihood.get_shape().as_list())
    return likelihood


def inference_multiview(views, n_classes, keep_prob):
    """
    views: N x V x W x H x C tensor
    """
    n_views = views.get_shape().as_list()[1] 

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])
    
    view_pool = []
    view_fc = []
    view_likelihood = []
    group_views = n_views // g_.NUM_GROUPS
    for i in range(n_views):
        view = tf.gather(views, i) # NxWxHxC

        if g_.NUM_GROUPS == 1:
            # set reuse True for i > 0, for weight-sharing
            reuse = (i != 0)

            conv1 = _conv('conv1_7x7_s2', view, [7, 7, 3, 64], [1, 2, 2, 1], reuse=reuse)
            pool1 = _maxpool('pool1_3x3_s2', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])
            lrn1 = _lrn('pool1_norm1', pool1)

            conv2_reduce = _conv('conv2_3x3_reduce', lrn1, [1, 1, 64, 64], reuse=reuse)
            conv2 = _conv('conv2_3x3', conv2_reduce, [3, 3, 64, 192], reuse=reuse)
            lrn2 = _lrn('conv2_norm2', conv2)
            pool2 = _maxpool('pool2_3x3_s2', lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])

            mixed3a_br0 = _conv('inception_3a_1x1', pool2, [1, 1, 192, 64], reuse=reuse)
            mixed3a_br1 = _conv('inception_3a_3x3_reduce', pool2, [1, 1, 192, 96], reuse=reuse)
            mixed3a_br1 = _conv('inception_3a_3x3', mixed3a_br1, [3, 3, 96, 128], reuse=reuse)
            mixed3a_br2 = _conv('inception_3a_5x5_reduce', pool2, [1, 1, 192, 16], reuse=reuse)
            mixed3a_br2 = _conv('inception_3a_5x5', mixed3a_br2, [5, 5, 16, 32], reuse=reuse)
            mixed3a_br3 = _maxpool('inception_3a_pool', pool2, [1, 3, 3, 1])
            mixed3a_br3 = _conv('inception_3a_pool_proj', mixed3a_br3, [1, 1, 192, 32], reuse=reuse)
            mixed3a = tf.concat([mixed3a_br0, mixed3a_br1, mixed3a_br2, mixed3a_br3], 3, name='inception_3a_output')

            mixed3b_br0 = _conv('inception_3b_1x1', mixed3a, [1, 1, 256, 128], reuse=reuse)
            mixed3b_br1 = _conv('inception_3b_3x3_reduce', mixed3a, [1, 1, 256, 128], reuse=reuse)
            mixed3b_br1 = _conv('inception_3b_3x3', mixed3b_br1, [3, 3, 128, 192], reuse=reuse)
            mixed3b_br2 = _conv('inception_3b_5x5_reduce', mixed3a, [1, 1, 256, 32], reuse=reuse)
            mixed3b_br2 = _conv('inception_3b_5x5', mixed3b_br2, [5, 5, 32, 96], reuse=reuse)
            mixed3b_br3 = _maxpool('inception_3b_pool', mixed3a, [1, 3, 3, 1])
            mixed3b_br3 = _conv('inception_3b_pool_proj', mixed3b_br3, [1, 1, 256, 64], reuse=reuse)
            mixed3b = tf.concat([mixed3b_br0, mixed3b_br1, mixed3b_br2, mixed3b_br3], 3, name='inception_3b_output')
            pool3 = _maxpool('pool3_3x3_s2', mixed3b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])

            mixed4a_br0 = _conv('inception_4a_1x1', pool3, [1, 1, 480, 192], reuse=reuse)
            mixed4a_br1 = _conv('inception_4a_3x3_reduce', pool3, [1, 1, 480, 96], reuse=reuse)
            mixed4a_br1 = _conv('inception_4a_3x3', mixed4a_br1, [3, 3, 96, 208], reuse=reuse)
            mixed4a_br2 = _conv('inception_4a_5x5_reduce', pool3, [1, 1, 480, 16], reuse=reuse)
            mixed4a_br2 = _conv('inception_4a_5x5', mixed4a_br2, [5, 5, 16, 48], reuse=reuse)
            mixed4a_br3 = _maxpool('inception_4a_pool', pool3, [1, 3, 3, 1])
            mixed4a_br3 = _conv('inception_4a_pool_proj', mixed4a_br3, [1, 1, 480, 64], reuse=reuse)
            mixed4a = tf.concat([mixed4a_br0, mixed4a_br1, mixed4a_br2, mixed4a_br3], 3, name='inception_4a_output')

            mixed4b_br0 = _conv('inception_4b_1x1', mixed4a, [1, 1, 512, 160], reuse=reuse)
            mixed4b_br1 = _conv('inception_4b_3x3_reduce', mixed4a, [1, 1, 512, 112], reuse=reuse)
            mixed4b_br1 = _conv('inception_4b_3x3', mixed4b_br1, [3, 3, 112, 224], reuse=reuse)
            mixed4b_br2 = _conv('inception_4b_5x5_reduce', mixed4a, [1, 1, 512, 24], reuse=reuse)
            mixed4b_br2 = _conv('inception_4b_5x5', mixed4b_br2, [5, 5, 24, 64], reuse=reuse)
            mixed4b_br3 = _maxpool('inception_4b_pool', mixed4a, [1, 3, 3, 1])
            mixed4b_br3 = _conv('inception_4b_pool_proj', mixed4b_br3, [1, 1, 512, 64], reuse=reuse)
            mixed4b = tf.concat([mixed4b_br0, mixed4b_br1, mixed4b_br2, mixed4b_br3], 3, name='inception_4b_output')
            mixed4c_br0 = _conv('inception_4c_1x1', mixed4b, [1, 1, 512, 128], reuse=reuse)
            mixed4c_br1 = _conv('inception_4c_3x3_reduce', mixed4b, [1, 1, 512, 128], reuse=reuse)
            mixed4c_br1 = _conv('inception_4c_3x3', mixed4c_br1, [3, 3, 128, 256], reuse=reuse)
            mixed4c_br2 = _conv('inception_4c_5x5_reduce', mixed4b, [1, 1, 512, 24], reuse=reuse)
            mixed4c_br2 = _conv('inception_4c_5x5', mixed4c_br2, [5, 5, 24, 64], reuse=reuse)
            mixed4c_br3 = _maxpool('inception_4c_pool', mixed4b, [1, 3, 3, 1])
            mixed4c_br3 = _conv('inception_4c_pool_proj', mixed4c_br3, [1, 1, 512, 64], reuse=reuse)
            mixed4c = tf.concat([mixed4c_br0, mixed4c_br1, mixed4c_br2, mixed4c_br3], 3, name='inception_4c_output')

            mixed4d_br0 = _conv('inception_4d_1x1', mixed4c, [1, 1, 512, 112], reuse=reuse)
            mixed4d_br1 = _conv('inception_4d_3x3_reduce', mixed4c, [1, 1, 512, 144], reuse=reuse)
            mixed4d_br1 = _conv('inception_4d_3x3', mixed4d_br1, [3, 3, 144, 288], reuse=reuse)
            mixed4d_br2 = _conv('inception_4d_5x5_reduce', mixed4c, [1, 1, 512, 32], reuse=reuse)
            mixed4d_br2 = _conv('inception_4d_5x5', mixed4d_br2, [5, 5, 32, 64], reuse=reuse)
            mixed4d_br3 = _maxpool('inception_4d_pool', mixed4c, [1, 3, 3, 1])
            mixed4d_br3 = _conv('inception_4d_pool_proj', mixed4d_br3, [1, 1, 512, 64], reuse=reuse)
            mixed4d = tf.concat([mixed4d_br0, mixed4d_br1, mixed4d_br2, mixed4d_br3], 3, name='inception_4d_output')

            ind = 0
        else:
            ind = i // group_views
            reuse = ((i % group_views) != 0)

            conv1 = _conv('vg%d_conv1_7x7_s2'%ind, view, [7, 7, 3, 64], [1, 2, 2, 1], reuse=reuse)
            pool1 = _maxpool('vg%d_pool1_3x3_s2'%ind, conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])
            lrn1 = _lrn('vg%d_pool1_norm1'%ind, pool1)

            conv2_reduce = _conv('vg%d_conv2_3x3_reduce'%ind, lrn1, [1, 1, 64, 64], reuse=reuse)
            conv2 = _conv('vg%d_conv2_3x3'%ind, conv2_reduce, [3, 3, 64, 192], reuse=reuse)
            lrn2 = _lrn('vg%d_conv2_norm2'%ind, conv2)
            pool2 = _maxpool('vg%d_pool2_3x3_s2'%ind, lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])

            mixed3a_br0 = _conv('vg%d_inception_3a_1x1'%ind, pool2, [1, 1, 192, 64], reuse=reuse)
            mixed3a_br1 = _conv('vg%d_inception_3a_3x3_reduce'%ind, pool2, [1, 1, 192, 96], reuse=reuse)
            mixed3a_br1 = _conv('vg%d_inception_3a_3x3'%ind, mixed3a_br1, [3, 3, 96, 128], reuse=reuse)
            mixed3a_br2 = _conv('vg%d_inception_3a_5x5_reduce'%ind, pool2, [1, 1, 192, 16], reuse=reuse)
            mixed3a_br2 = _conv('vg%d_inception_3a_5x5'%ind, mixed3a_br2, [5, 5, 16, 32], reuse=reuse)
            mixed3a_br3 = _maxpool('vg%d_inception_3a_pool'%ind, pool2, [1, 3, 3, 1])
            mixed3a_br3 = _conv('vg%d_inception_3a_pool_proj'%ind, mixed3a_br3, [1, 1, 192, 32], reuse=reuse)
            mixed3a = tf.concat([mixed3a_br0, mixed3a_br1, mixed3a_br2, mixed3a_br3], 3, name='vg%d_inception_3a_output'%ind)

            mixed3b_br0 = _conv('vg%d_inception_3b_1x1'%ind, mixed3a, [1, 1, 256, 128], reuse=reuse)
            mixed3b_br1 = _conv('vg%d_inception_3b_3x3_reduce'%ind, mixed3a, [1, 1, 256, 128], reuse=reuse)
            mixed3b_br1 = _conv('vg%d_inception_3b_3x3'%ind, mixed3b_br1, [3, 3, 128, 192], reuse=reuse)
            mixed3b_br2 = _conv('vg%d_inception_3b_5x5_reduce'%ind, mixed3a, [1, 1, 256, 32], reuse=reuse)
            mixed3b_br2 = _conv('vg%d_inception_3b_5x5'%ind, mixed3b_br2, [5, 5, 32, 96], reuse=reuse)
            mixed3b_br3 = _maxpool('vg%d_inception_3b_pool'%ind, mixed3a, [1, 3, 3, 1])
            mixed3b_br3 = _conv('vg%d_inception_3b_pool_proj'%ind, mixed3b_br3, [1, 1, 256, 64], reuse=reuse)
            mixed3b = tf.concat([mixed3b_br0, mixed3b_br1, mixed3b_br2, mixed3b_br3], 3, name='vg%d_inception_3b_output'%ind)
            pool3 = _maxpool('vg%d_pool3_3x3_s2'%ind, mixed3b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])

            mixed4a_br0 = _conv('vg%d_inception_4a_1x1'%ind, pool3, [1, 1, 480, 192], reuse=reuse)
            mixed4a_br1 = _conv('vg%d_inception_4a_3x3_reduce'%ind, pool3, [1, 1, 480, 96], reuse=reuse)
            mixed4a_br1 = _conv('vg%d_inception_4a_3x3'%ind, mixed4a_br1, [3, 3, 96, 208], reuse=reuse)
            mixed4a_br2 = _conv('vg%d_inception_4a_5x5_reduce'%ind, pool3, [1, 1, 480, 16], reuse=reuse)
            mixed4a_br2 = _conv('vg%d_inception_4a_5x5'%ind, mixed4a_br2, [5, 5, 16, 48], reuse=reuse)
            mixed4a_br3 = _maxpool('vg%d_inception_4a_pool'%ind, pool3, [1, 3, 3, 1])
            mixed4a_br3 = _conv('vg%d_inception_4a_pool_proj'%ind, mixed4a_br3, [1, 1, 480, 64], reuse=reuse)
            mixed4a = tf.concat([mixed4a_br0, mixed4a_br1, mixed4a_br2, mixed4a_br3], 3, name='vg%d_inception_4a_output'%ind)

            mixed4b_br0 = _conv('vg%d_inception_4b_1x1'%ind, mixed4a, [1, 1, 512, 160], reuse=reuse)
            mixed4b_br1 = _conv('vg%d_inception_4b_3x3_reduce'%ind, mixed4a, [1, 1, 512, 112], reuse=reuse)
            mixed4b_br1 = _conv('vg%d_inception_4b_3x3'%ind, mixed4b_br1, [3, 3, 112, 224], reuse=reuse)
            mixed4b_br2 = _conv('vg%d_inception_4b_5x5_reduce'%ind, mixed4a, [1, 1, 512, 24], reuse=reuse)
            mixed4b_br2 = _conv('vg%d_inception_4b_5x5'%ind, mixed4b_br2, [5, 5, 24, 64], reuse=reuse)
            mixed4b_br3 = _maxpool('vg%d_inception_4b_pool'%ind, mixed4a, [1, 3, 3, 1])
            mixed4b_br3 = _conv('vg%d_inception_4b_pool_proj'%ind, mixed4b_br3, [1, 1, 512, 64], reuse=reuse)
            mixed4b = tf.concat([mixed4b_br0, mixed4b_br1, mixed4b_br2, mixed4b_br3], 3, name='vg%d_inception_4b_output'%ind)
            mixed4c_br0 = _conv('vg%d_inception_4c_1x1'%ind, mixed4b, [1, 1, 512, 128], reuse=reuse)
            mixed4c_br1 = _conv('vg%d_inception_4c_3x3_reduce'%ind, mixed4b, [1, 1, 512, 128], reuse=reuse)
            mixed4c_br1 = _conv('vg%d_inception_4c_3x3'%ind, mixed4c_br1, [3, 3, 128, 256], reuse=reuse)
            mixed4c_br2 = _conv('vg%d_inception_4c_5x5_reduce'%ind, mixed4b, [1, 1, 512, 24], reuse=reuse)
            mixed4c_br2 = _conv('vg%d_inception_4c_5x5'%ind, mixed4c_br2, [5, 5, 24, 64], reuse=reuse)
            mixed4c_br3 = _maxpool('vg%d_inception_4c_pool'%ind, mixed4b, [1, 3, 3, 1])
            mixed4c_br3 = _conv('vg%d_inception_4c_pool_proj'%ind, mixed4c_br3, [1, 1, 512, 64], reuse=reuse)
            mixed4c = tf.concat([mixed4c_br0, mixed4c_br1, mixed4c_br2, mixed4c_br3], 3, name='vg%d_inception_4c_output'%ind)

            mixed4d_br0 = _conv('vg%d_inception_4d_1x1'%ind, mixed4c, [1, 1, 512, 112], reuse=reuse)
            mixed4d_br1 = _conv('vg%d_inception_4d_3x3_reduce'%ind, mixed4c, [1, 1, 512, 144], reuse=reuse)
            mixed4d_br1 = _conv('vg%d_inception_4d_3x3'%ind, mixed4d_br1, [3, 3, 144, 288], reuse=reuse)
            mixed4d_br2 = _conv('vg%d_inception_4d_5x5_reduce'%ind, mixed4c, [1, 1, 512, 32], reuse=reuse)
            mixed4d_br2 = _conv('vg%d_inception_4d_5x5'%ind, mixed4d_br2, [5, 5, 32, 64], reuse=reuse)
            mixed4d_br3 = _maxpool('vg%d_inception_4d_pool'%ind, mixed4c, [1, 3, 3, 1])
            mixed4d_br3 = _conv('vg%d_inception_4d_pool_proj'%ind, mixed4d_br3, [1, 1, 512, 64], reuse=reuse)
            mixed4d = tf.concat([mixed4d_br0, mixed4d_br1, mixed4d_br2, mixed4d_br3], 3, name='vg%d_inception_4d_output'%ind)

        if g_.VIEWPOOL == 'wavg':
            vg_pool = _avgpool('vg%02d_pool'%ind, mixed4d, [1, 5, 5, 1], strides=[1, 3, 3, 1])
            vg_conv = _conv('vg%02d_conv'%ind, vg_pool, [1, 1, 528, 128], reuse=reuse)
            dim = np.prod(vg_conv.get_shape().as_list()[1:])
            vg_fc = _fc('vg%02d_fc'%ind, tf.reshape(vg_conv, [-1, dim]), 1024, reuse=reuse)
            vg_fc = _fc('vg%02d_classifier'%ind, vg_fc, n_classes, reuse=reuse)
            vp_likelihood = _softmax_likelihood('vp%02d_softmax_likelihood'%i, vg_fc, threshold=FLAGS.likelihood_threshold)

            dim = np.prod(vg_fc.get_shape().as_list()[1:])
            view_fc.append(tf.reshape(vg_fc, [-1, dim]))
            view_likelihood.append(tf.reshape(vp_likelihood, [-1, 1]))

        view_shape = mixed4d.get_shape().as_list()
        dim = np.prod(view_shape[1:])
        reshape = tf.reshape(mixed4d, [-1, dim])
        
        view_pool.append(reshape)

    try:
        if g_.VIEWPOOL == 'max':
            pool_vp = _view_pool(view_pool, 'pool_vp')
            print('pool_vp', pool_vp.get_shape().as_list())
        elif g_.VIEWPOOL == 'avg':
            pool_vp = _view_avgpool(view_pool, 'pool_avgvp')
            print('pool_avgvp', pool_vp.get_shape().as_list())
        elif g_.VIEWPOOL == 'wavg':
            pool_vp = _view_wavgpool(view_pool, view_likelihood, 'pool_wavgvp')
            print('pool_wavgvp', pool_vp.get_shape().as_list())
        else:
            raise ValueError
    except ValueError:
        print('Undefined option in view pooling')
        raise

    pool_vp = tf.reshape(pool_vp, [s if s is not None else -1 for s in view_shape])

    mixed4e_br0 = _conv('inception_4e_1x1', pool_vp, [1, 1, view_shape[-1], 256])
    mixed4e_br1 = _conv('inception_4e_3x3_reduce', pool_vp, [1, 1, view_shape[-1], 160])
    mixed4e_br1 = _conv('inception_4e_3x3', mixed4e_br1, [3, 3, 160, 320])
    mixed4e_br2 = _conv('inception_4e_5x5_reduce', pool_vp, [1, 1, view_shape[-1], 32])
    mixed4e_br2 = _conv('inception_4e_5x5', mixed4e_br2, [5, 5, 32, 128])
    mixed4e_br3 = _maxpool('inception_4e_pool', pool_vp, [1, 3, 3, 1])
    mixed4e_br3 = _conv('inception_4e_pool_proj', mixed4e_br3, [1, 1, view_shape[-1], 128])
    mixed4e = tf.concat([mixed4e_br0, mixed4e_br1, mixed4e_br2, mixed4e_br3], 3, name='inception_4e_output')
    pool4 = _maxpool('pool4_3x3_s2', mixed4e, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])

    mixed5a_br0 = _conv('inception_5a_1x1', pool4, [1, 1, 832, 256])
    mixed5a_br1 = _conv('inception_5a_3x3_reduce', pool4, [1, 1, 832, 160])
    mixed5a_br1 = _conv('inception_5a_3x3', mixed5a_br1, [3, 3, 160, 320])
    mixed5a_br2 = _conv('inception_5a_5x5_reduce', pool4, [1, 1, 832, 32])
    mixed5a_br2 = _conv('inception_5a_5x5', mixed5a_br2, [5, 5, 32, 128])
    mixed5a_br3 = _maxpool('inception_5a_pool', pool4, [1, 3, 3, 1])
    mixed5a_br3 = _conv('inception_5a_pool_proj', mixed5a_br3, [1, 1, 832, 128])
    mixed5a = tf.concat([mixed5a_br0, mixed5a_br1, mixed5a_br2, mixed5a_br3], 3, name='inception_5a_output')

    mixed5b_br0 = _conv('inception_5b_1x1', mixed5a, [1, 1, 832, 384])
    mixed5b_br1 = _conv('inception_5b_3x3_reduce', mixed5a, [1, 1, 832, 192])
    mixed5b_br1 = _conv('inception_5b_3x3', mixed5b_br1, [3, 3, 192, 384])
    mixed5b_br2 = _conv('inception_5b_5x5_reduce', mixed5a, [1, 1, 832, 48])
    mixed5b_br2 = _conv('inception_5b_5x5', mixed5b_br2, [5, 5, 48, 128])
    mixed5b_br3 = _maxpool('inception_5b_pool', mixed5a, [1, 3, 3, 1])
    mixed5b_br3 = _conv('inception_5b_pool_proj', mixed5b_br3, [1, 1, 832, 128])
    mixed5b = tf.concat([mixed5b_br0, mixed5b_br1, mixed5b_br2, mixed5b_br3], 3, name='inception_5b_output')
    pool5 = _avgpool('pool5_7x7_s1', mixed5b, [1, 7, 7, 1], padding='VALID')

    dim = np.prod(pool5.get_shape().as_list()[1:])
    drop = tf.nn.dropout(pool5, keep_prob)
    fc = _fc('classifier', tf.reshape(drop, [-1, dim]), n_classes)

    return fc, view_fc
    

def load_alexnet_to_mvcnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath, encoding = 'latin1')
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        if g_.NUM_GROUPS == 1:
            name = l
            _load_param(sess, name, data_dict[l])
        else:
            for i in range(g_.NUM_GROUPS):
                name = 'vg%d_'%i + l
                _load_param(sess, name, data_dict[l])

    for l in ['fc6', 'fc7']:
        name = l
        _load_param(sess, name, data_dict[l])
    
def load_googlenet_to_mvcnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath, encoding = 'latin1')
    data_dict = caffemodel.item()
    layers = ['conv1_7x7_s2', 'conv2_3x3_reduce', 'conv2_3x3'];
    for l in ['3a', '3b', '4a', '4b', '4c', '4d']:
        layers.append('inception_%s_1x1'%l)
        layers.append('inception_%s_3x3_reduce'%l)
        layers.append('inception_%s_3x3'%l)
        layers.append('inception_%s_5x5_reduce'%l)
        layers.append('inception_%s_5x5'%l)
        layers.append('inception_%s_pool_proj'%l)

    for l in layers:
        if g_.NUM_GROUPS == 1:
            name = l
            _load_param(sess, name, data_dict[l])
        else:
            for i in range(g_.NUM_GROUPS):
                name = 'vg%d_'%i + l
                _load_param(sess, name, data_dict[l])

    layers = []
    for l in ['4e', '5a', '5b']:
        layers.append('inception_%s_1x1'%l)
        layers.append('inception_%s_3x3_reduce'%l)
        layers.append('inception_%s_3x3'%l)
        layers.append('inception_%s_5x5_reduce'%l)
        layers.append('inception_%s_5x5'%l)
        layers.append('inception_%s_pool_proj'%l)

    for l in layers:
        name = l
        _load_param(sess, name, data_dict[l])
 
def _load_param(sess, name, layer_data):
    w, b = layer_data.values()

    with tf.variable_scope(name, reuse=True):
        for subkey, data in list(zip(('weights', 'biases'), (w, b))):
            print('loading ', name, subkey)

            try:
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))
            except ValueError as e: 
                print('varirable loading failed:', subkey, '(%s)' % str(e))


def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    print('vp before reducing:', vp.get_shape().as_list())
    vp = tf.reduce_max(vp, [0], name=name)
    return vp 

def _view_avgpool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    print('avg vp before reducing:', vp.get_shape().as_list())
    vp = tf.reduce_mean(vp, [0], name=name)
    return vp 

def _view_wavgpool(view_features, weights, name):
    wp = tf.expand_dims(weights[0], 0)
    for vw in weights[1:]:
        vw = tf.expand_dims(vw, 0)
        wp = tf.concat([wp, vw], 0)
    print('wavg wp before reducing:', wp.get_shape().as_list())
    wp = tf.reciprocal(tf.reduce_sum(wp, [0]))

    for ind, v in enumerate(view_features):
        w = tf.reshape(tf.multiply(weights[ind], wp), [-1])
        if FLAGS.likelihood_threshold > 0.0 and g_.DEBUG == True:
            w = tf.Print(w, [tf.count_nonzero(w)], 'view #%d actvation: ' % (ind))
        v = tf.expand_dims(tf.multiply(v, w[:, tf.newaxis]), 0) # eg. [100] -> [1, 100]
        if ind == 0:
            vp = v
        else:
            vp = tf.concat([vp, v], 0)
    print('wavg vp before reducing:', vp.get_shape().as_list())
    vp = tf.reduce_sum(vp, [0], name=name)
    return vp

def loss(fc8, labels, name='total_loss'):
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc8)
    l = tf.reduce_mean(l)
    
    tf.add_to_collection('losses', l)

    return tf.add_n(tf.get_collection('losses'), name=name)


def classify(fc8):
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    print('losses:', losses)
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
    

def train(total_loss, global_step, data_size):
    num_batches_per_epoch = data_size / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        if FLAGS.clip_gradients > 0.0:
            grads = [(tf.clip_by_norm(grad, FLAGS.clip_gradients), var) for grad, var in grads]
    
    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
