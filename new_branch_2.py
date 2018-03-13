import tensorflow as tf
import numpy as np
dim =4096
lstmnum=3
lstmround=8

def maxout(inputs, num_units=dim, axis=None):
    shape = inputs.get_shape().as_list()
    if axis is None:
        # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'
                         .format(num_channels, num_units))
    shape[axis] = -1
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return tf.nn.sigmoid(outputs)


def memory(stat, input1, last, scope):
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        ft = tf.contrib.layers.fully_connected(stat, dim, activation_fn=tf.nn.sigmoid,
                                              normalizer_fn=tf.contrib.layers.batch_norm, scope='3lun' + 'memoryft1' + scope) + tf.contrib.layers.fully_connected(
            input1, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm,scope='3lun' + 'memoryft2' + scope)
        it = tf.contrib.layers.fully_connected(stat, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm,
                                               scope='3lun' + 'memoryit1' + scope) + tf.contrib.layers.fully_connected(
            input1, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope='3lun' + 'memoryit2' + scope)
        ct = tf.contrib.layers.fully_connected(stat, dim, activation_fn=tf.nn.tanh,normalizer_fn=tf.contrib.layers.batch_norm,
                                               scope='3lun' + 'memoryct1' + scope) + tf.contrib.layers.fully_connected(
            input1, dim, activation_fn=tf.nn.tanh,normalizer_fn=tf.contrib.layers.batch_norm, scope='3lun' + 'memoryct2' + scope)
        ctnew = tf.multiply(ft, last) + tf.multiply(it, ct)
        ot = tf.contrib.layers.fully_connected(stat, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm,
                                               scope='3lun' + 'memoryot1' + scope) + tf.contrib.layers.fully_connected(
            input1, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope='3lun' + 'memoryot2' + scope)
        htnew = tf.multiply(ot, tf.nn.tanh(ctnew))
        return htnew, ctnew


def dropout(tensor, is_train=0):
    if is_train == 1:
        keep_prob = 0.5
    else:
        keep_prob = 1.0
    return tf.nn.dropout(tensor, keep_prob)


def usememory(stat,input1,last,scope):
    ht, ct = memory(stat, input1, last, scope)
    return ht,ct

def lstm(stat,input1,last,scope):
    for i in range(lstmnum):
        stat[i],last=usememory(stat[i],input1,last,scope+str(i))
    return stat,last

def model(inputx, batchsize, is_train):  # normalizer_fn=tf.contrib.layers.batch_norm,
    with tf.name_scope('3lun') as scope:
        ht=list(np.zeros(lstmnum))
        fc1 = dropout(tf.contrib.layers.fully_connected(inputx, dim, activation_fn=tf.nn.elu,normalizer_fn=tf.contrib.layers.batch_norm,
                                                         scope=scope + '1'),
                      is_train)
        fc2 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx,fc1],1), dim, activation_fn=tf.nn.elu,normalizer_fn=tf.contrib.layers.batch_norm,
                                                         scope=scope + '2'),
                      is_train)

        fc3 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx,fc2],1), dim, activation_fn=tf.nn.elu,normalizer_fn=tf.contrib.layers.batch_norm,

                                                         scope=scope + '3'),
                      is_train)
        fc4 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx,fc3],1), dim, activation_fn=tf.nn.elu,
normalizer_fn=tf.contrib.layers.batch_norm,                                                         scope=scope + '4'),
                      is_train)
        fc5 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx,fc4],1), dim, activation_fn=tf.nn.elu,
                                           normalizer_fn=tf.contrib.layers.batch_norm,              scope=scope + '5'),
                      is_train)

        fc6 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx,fc5],1), dim, activation_fn=tf.nn.elu,normalizer_fn=tf.contrib.layers.batch_norm,
                                                         scope=scope + '6'),
                      is_train)

        fc7 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx,fc6],1), dim, activation_fn=tf.nn.elu,normalizer_fn=tf.contrib.layers.batch_norm,
                                                         scope=scope + '7'),
                      is_train)

        input1 = inputx
        for i in range(lstmnum):
            ht[i]=fc7
        last = fc1
        ht, ct = lstm(ht,input1,last,'lstm')
        for i in range(lstmround):
            ht, ct = lstm(ht, input1, ct, 'lstm')
        

        fc35 = tf.contrib.layers.fully_connected(tf.concat([inputx,ct],1), 2, activation_fn=tf.nn.elu,
                                                 normalizer_fn=tf.contrib.layers.batch_norm,
 scope=scope + '35')
        out = tf.nn.softmax(fc35)

        return fc35

