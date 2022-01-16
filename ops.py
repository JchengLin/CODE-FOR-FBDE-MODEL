import math
import numpy as np
from utils import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib as tf_contrib

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', 
                            use_bias=True, sn=False, reuse=False, scope='conv_0'):
    with tf.variable_scope(scope, reuse=reuse):
        if pad > 0 :
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], 
                              [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], 
                              [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], 
                                initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)
        return x

def fully_connected_with_w(x, use_bias=True, sn=False, reuse=False, scope='linear'):
    with tf.variable_scope(scope, reuse=reuse):
        x = flatten(x)
        bias = 0.0
        shape = x.get_shape().as_list()
        channels = shape[-1]

        w = tf.get_variable("kernel", [channels, 1], tf.float32,
                            initializer=weight_init, regularizer=weight_regularizer)

        if sn :
            w = spectral_norm(w)

        if use_bias :
            bias = tf.get_variable("bias", [1],
                                   initializer=tf.constant_initializer(0.0))

            x = tf.matmul(x, w) + bias
        else :
            x = tf.matmul(x, w)

        if use_bias :
            weights = tf.gather(tf.transpose(tf.nn.bias_add(w, bias)), 0)
        else :
            weights = tf.gather(tf.transpose(w), 0)

        return x, weights

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, 
                                kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x
        
##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock_0'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x_init + x

def adaptive_ins_layer_resblock(x_init, channels, gamma, beta, use_bias=True, smoothing=True, scope='adaptive_resblock') :
                                
    with tf.variable_scope(scope):
    
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_layer_norm(x, gamma, beta, smoothing)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_layer_norm(x, gamma, beta, smoothing)

        return x_init + x
                             
def psa_block(x_init, scope='polarized_self_attention_block'):

    with tf.variable_scope(scope):

        context_channel = spatial_pool(x_init)
        context_spatial = channel_pool(x_init)
        out = tf.add(context_channel, context_spatial)

    return out

def spatial_pool(x_init, ratio=4, scope='spatial_attention'):

    with tf.variable_scope(scope):

        batch, height, width, channels = x_init.get_shape().as_list()
        assert channels % 2 == 0
        channel = channels // 2
        input_x = tf.layers.conv2d(inputs=x_init, filters=channel, kernel_size=1,
                                 strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)
        input_x = tf.reshape(input_x, [width * height, channel])
    
        context_mask = tf.layers.conv2d(inputs=x_init, filters=1, kernel_size=1,
                                 strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)
        context_mask = tf.reshape(context_mask, [width * height, 1])
        context_mask = tf.nn.softmax(context_mask, axis=1)
        context = tf.matmul(input_x, context_mask, transpose_a=True)
        context = tf.transpose(context, [1, 0])
        context = tf.reshape(context, [1, 1, 1, channel])
        
        context = tf.layers.conv2d(inputs=x_init, filters=channels, kernel_size=1, strides=1, padding='same')
        print(context)
    
        mask_ch = tf.nn.sigmoid(context)
    
    return tf.multiply(x_init, mask_ch)


def channel_pool(x_init, scope='channel_attention'):

    with tf.variable_scope(scope):
        
        batch, height, width, channels = x_init.get_shape().as_list()
        assert channels % 2 == 0
        channel = channels // 2
        g_x = tf.layers.conv2d(inputs=x_init, filters=channel, kernel_size=1,
                                 strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)
        avg_x = global_avg_pooling(g_x, keepdims=True)
        avg_x = tf.nn.softmax(avg_x)
        avg_x = tf.reshape(avg_x, [channel, 1])
            
        theta_x = tf.layers.conv2d(inputs=x_init, filters=channel, kernel_size=1,
                                 strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)
        theta_x = tf.reshape(theta_x, [height * width, channel])
        context = tf.matmul(theta_x, avg_x)
        
        context = tf.reshape(context, [height * width,])
        mask_sp = tf.nn.sigmoid(context)
        mask_sp = tf.reshape(mask_sp, [height, width, 1])

    return tf.multiply(x_init, mask_sp)
    
##################################################################################
# Sampling
##################################################################################

def down_sample(x) :
    return avg_pooling(x, kernel=3, stride=2, pad=1)
    
def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size   = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def global_avg_pooling(x, keepdims=False):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=keepdims)
    return gap

def global_max_pooling(x, keepdims=False):
    gmp = tf.reduce_max(x, axis=[1, 2], keepdims=keepdims)
    return gmp
    
def max_pooling(x) :
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')

def avg_pooling(x, kernel=2, stride=2, pad=0) :
    if pad > 0 :
        if (kernel - stride) % 2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride, padding='VALID')
            
##################################################################################
# Activation function
##################################################################################

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

def grelu(x):
    return 0.5*x*(1+tf.math.tanh(tf.math.sqrt(2/math.pi) * (x+0.044715*tf.math.pow(x, 3))))
    
def tanh(x):
    return tf.tanh(x)

def sigmoid(x) :
    return tf.sigmoid(x)

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def l2_norm(x):

    return x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 1e-10)
    
##################################################################################
# Normalization function
##################################################################################

def adaptive_instance_layer_norm(x, gamma, beta, smoothing=True, scope='instance_layer_norm') :
    with tf.variable_scope(scope):
        ch  = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho  = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0), constraint=lambda x:\
               tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

        if smoothing :
            rho = tf.clip_by_value(rho - tf.constant(0.1), 0.0, 1.0)

        x_hat   = rho * x_ins + (1 - rho) * x_ln
        x_hat   = x_hat * gamma + beta

        return x_hat
        
def batch_norm(x, scope="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=False, scope=scope)

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta
    
def instance_norm(x, scope='instance_norm') :
    return tf_contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True, scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x, center=True, scale=True, scope=scope)

def layer_instance_norm(x, scope='layer_instance_norm') :
    with tf.variable_scope(scope):
        ch  = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
        x_ln  = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho   = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(0.0), \
              constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta  = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_ins + (1 - rho) * x_ln

        x_hat = x_hat * gamma + beta

        return x_hat

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):    
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################
        
def reg_loss(scope_name):
    '''Compute regularization loss for model.'''
    
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization:
        if scope_name in item.name :
            loss.append(item)

    return tf.reduce_sum(loss)

def L1_loss(x, y):
    '''Compute l1 loss for model.'''
    
    return tf.reduce_mean(tf.abs(x - y))
    
def contrastive_loss(feat_s, feat_t, temperature=1.0):
    """ Compute the contrastive_loss bettwen the realA and fakeB images loss for model. """
    
    numpatchs, dim = feat_s.shape.as_list()
    logit = tf.matmul(feat_s, feat_t, transpose_b=True) / temperature
    
    # Diagonal entries are pos logits, the others are neg logits.
    label = tf.eye(numpatchs, dtype=tf.float32)
    
    loss = tf.losses.softmax_cross_entropy(label, logit)

    return loss

def histogram_loss(x, y):
    histogram_x = get_histogram(x)
    histogram_y = get_histogram(y)

    hist_loss = L1_loss(histogram_x, histogram_y)

    return hist_loss

def normalization(x):
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
    
    return x

def get_histogram(img, bin_size=0.2):
    hist_entries = []

    img_r, img_g, img_b = tf.split(img, num_or_size_splits=3, axis=-1)

    for img_chan in [img_r, img_g, img_b]:
        for i in np.arange(-1, 1, bin_size):
            gt = tf.greater(img_chan, i)
            leq = tf.less_equal(img_chan, i + bin_size)

            condition = tf.cast(tf.logical_and(gt, leq), tf.float32)
            hist_entries.append(tf.reduce_sum(condition))

    hist = normalization(hist_entries)

    return hist
    
def dloss(loss_func, real, fake, m=0.1):
    """Compute discriminative loss for model."""
    
    loss = []
    real_loss = 0
    fake_loss = 0
    for i in range(len(real)):

        if loss_func == 'lsgan' :
            real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake[i]))
            adv_loss = real_loss + fake_loss

        if loss_func == 'pulsgan' :
            labels = tf.ones_like(real[i])
            d_real = tf.reduce_mean(tf.square(real[i] - labels))
            d_post = tf.reduce_mean(tf.square(fake[i] + labels))
            d_nega = tf.reduce_mean(tf.square(real[i] + labels))
            d_fake = tf.maximum(d_post - m * d_nega, 0.0)
            adv_loss = d_fake + m * d_real

        if loss_func == 'lscut' :
            # (A, P) --> d(A, P) ----------------> dis_post.
            dis_post = tf.reduce_mean(tf.squared_difference(1.0, real[i]))
            # (A, N) --> d(A, N) ----------------> dis_negs.
            dis_negs = tf.reduce_mean(tf.squared_difference(1.0, fake[i]))
            # max(dis_post - m*dis_negs, 0.0) --> real loss.
            dis_real = tf.maximum(0.0, dis_post - m * dis_negs)
            # Lsgan loss -----------------------> fake loss.
            dis_fake = tf.reduce_mean(tf.squared_difference(0.0, fake[i]))
            # Lsgan loss ------> Triplts discriminator loss.
            adv_loss = (dis_real + dis_fake)
            
        if loss_func == 'softplus':
            real_loss = tf.reduce_mean(tf.math.log(1 + tf.math.exp(-real[i])))
            fake_loss = tf.reduce_mean(tf.math.log(1 + tf.math.exp(fake[i])))
            adv_loss = real_loss + fake_loss
            
        loss.append(adv_loss)

    return sum(loss)
    
def gloss(loss_func, fake):
    """Compute generator loss for model."""
    
    loss = []
    fake_loss = 0

    for i in range(len(fake)):

        if loss_func == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

        if loss_func == 'pulsgan' :
            labels= tf.ones_like(fake[i])
            fake_loss = tf.reduce_mean(tf.square(fake[i] - labels))

        if loss_func == 'lscut' :
            fake_loss = tf.reduce_mean(tf.squared_difference(1.0, fake[i]))
            
        if loss_func == 'softplus':
            fake_loss = tf.reduce_mean(tf.math.log(1 + tf.math.exp(-fake[i])))
                        
        loss.append(fake_loss)

    return sum(loss)