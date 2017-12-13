import tensorflow as tf
dim=4096


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
def memroy(stat,input,last,scope):
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
    	ft=tf.contrib.layers.fully_connected(stat,dim,activation_fn=tf.nn.sigmoid, scope='3lun' + 'memoryft1'+scope)+tf.contrib.layers.fully_connected(input, dim, activation_fn=tf.nn.sigmoid,scope='3lun' + 'memoryft2' + scope)
    	it=tf.contrib.layers.fully_connected(stat,dim,activation_fn=tf.nn.sigmoid,scope='3lun' + 'memoryit1'+scope)+tf.contrib.layers.fully_connected(input,dim,activation_fn=tf.nn.sigmoid, scope='3lun' + 'memoryit2'+scope)
    	ct=tf.contrib.layers.fully_connected(stat,dim,activation_fn=tf.nn.tanh,scope='3lun' + 'memoryct1'+scope)+tf.contrib.layers.fully_connected(input,dim,activation_fn=tf.nn.tanh,scope='3lun' + 'memoryct2'+scope)
    	ctnew=tf.multiply(ft,last)+tf.multiply(it,ct)
    	ot=tf.contrib.layers.fully_connected(stat,dim,activation_fn=tf.nn.sigmoid,scope='3lun' + 'memoryot1'+scope)+tf.contrib.layers.fully_connected(input,dim,activation_fn=tf.nn.sigmoid,scope='3lun' + 'memoryot2'+scope)
    	htnew=tf.multiply(ot,tf.nn.tanh(ctnew))
    	return htnew,ctnew
def dropout(tensor,is_train=0):
    if is_train==1:
        keep_prob=0.8
    else:
        keep_prob=1.0
    return tf.nn.dropout(tensor,keep_prob)

def model(inputx,batchsize,is_train):#normalizer_fn=tf.contrib.layers.batch_norm,
    with tf.name_scope('3lun') as scope:
        fc2 = dropout(tf.contrib.layers.fully_connected(inputx, dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '2'),is_train)

        fc3 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx,fc2],1), dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '3'),is_train)

        fc4 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc2,fc3],1), dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '4'),is_train)

        input = fc4
        stat = fc2
        last = fc3
        ht, ct = memroy(stat, input, last, 'fc')

        fc5 = dropout(tf.contrib.layers.fully_connected(ct, dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '5'),is_train)

        fc6 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc5],1), dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + '6'),is_train)

        fc7 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc5,fc6],1), dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '7'),is_train)

        input = fc7
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc8 = dropout(tf.contrib.layers.fully_connected(ct, dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + '8'),is_train)

        fc9 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc8],1), dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '9'),is_train)

        fc10 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc8,fc9],1), dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '10'),is_train)

        input = fc10
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc11 = dropout(tf.contrib.layers.fully_connected(ct, dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + '11'),is_train)



        fc12 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc11],1), dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '12') ,is_train) # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc13 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc11,fc12],1),dim,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + '13'),is_train)

        input = fc13
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc14 = dropout(tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '14'),is_train)

        fc15 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc14],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '15') ,is_train) # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc16 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc14,fc15],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + '16'),is_train)

        input = fc16
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc17 = dropout(tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '17'),is_train)

        fc18 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc17],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '18'),is_train)  # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc19 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc17,fc18],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '19'),is_train)

        input = fc19
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc20 = dropout(tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '20'),is_train)

        fc21 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc20],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '21'),is_train ) # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc22 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc20,fc21],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '22'),is_train)

        input = fc22
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')
	
	

        fc23 = dropout(tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '23'),is_train)

        fc24 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc23],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '24'),is_train)  # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc25 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc23,fc24],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '25'),is_train)

        input = fc25
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc26 = dropout(tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '26'),is_train)

        fc27 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc26],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '27'),is_train ) # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc28 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc26,fc27],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '28'),is_train)

        input = fc28
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc29 = dropout(tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '29'),is_train)

        fc30 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc29],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '30'),is_train)  # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc31 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc29,fc30],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '31'),is_train)

        input = fc31
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc32 = dropout(tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '32'),is_train)

        fc33 = dropout(tf.contrib.layers.fully_connected(tf.concat([ct,fc32],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '33'),is_train ) # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc34 = dropout(tf.contrib.layers.fully_connected(tf.concat([fc32,fc33],1), dim, activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '34'),is_train)

        input = fc34
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last, 'fc')

        fc35 = tf.contrib.layers.fully_connected(ct, 2,activation_fn=tf.nn.sigmoid,normalizer_fn=tf.contrib.layers.batch_norm, scope=scope + '35')
        out=tf.nn.softmax(fc35)



        return out
'''
    global forward1,forward2
    numoutput=512
    with tf.name_scope('aes') as scope:
        fc0 = tf.contrib.layers.fully_connected(inputx, numoutput, normalizer_fn=tf.contrib.layers.batch_norm,
                                                scope=scope+'0')  # ,normalizer_fn=tf.contrib.layers.batch_norm
        fc1 = tf.contrib.layers.fully_connected(fc0,numoutput,normalizer_fn=tf.contrib.layers.batch_norm,
                                                scope=scope+'1')
        forward1=fc0
        forward2=fc1
        for i in range(2,22):
            f=tf.contrib.layers.fully_connected(forward1+(forward2),numoutput,normalizer_fn=tf.contrib.layers.batch_norm,
                                                scope=scope+str(i))
            forward2=forward1
            forward1=f

        fin=tf.contrib.layers.fully_connected(forward1+(forward2),16,normalizer_fn=tf.contrib.layers.batch_norm,
                                                scope=scope+'fin')
        return fin
'''

'''
        fc2 = tf.contrib.layers.fully_connected(fc1, 1024*2, scope=scope + '2')

        fc3 = tf.contrib.layers.fully_connected(fc2, 1024*2, scope=scope + '3')

        fc4 = tf.contrib.layers.fully_connected(fc3, 1024*2, scope=scope + '4')

        fcc1 = tf.concat([fc2, fc3, fc4],1)

        fc5 = tf.contrib.layers.fully_connected(fcc1, 1024*2*2, scope=scope + '5')

        fc6 = tf.contrib.layers.fully_connected(fc5, 1024*2*2, scope=scope + '6')

        fc7 = tf.contrib.layers.fully_connected(fc6, 1024*2*2, scope=scope + '7')

        fcc2 = tf.concat([fc5, fc6, fc7],1)

        fc8 = tf.contrib.layers.fully_connected(fcc2, 1024*2 , scope=scope + '8')

        fc9 = tf.contrib.layers.fully_connected(fc8, 1024*2 , scope=scope + '9')

        fc10 = tf.contrib.layers.fully_connected(fc9, 1024*2 , scope=scope + '10')

        fc11 = tf.contrib.layers.fully_connected(fc10, 1024*2 , scope=scope + '11')

        fcc3 =tf.concat([fc8, fc9, fc10,fc11],1)

        fc12 = tf.contrib.layers.fully_connected(fcc3, 1024*2 ,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                scope=scope + '12')  # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc13 = tf.contrib.layers.fully_connected(fc12, 1024*2 , scope=scope + '13')

        fc14 = tf.contrib.layers.fully_connected(fc13, 1024*2 , scope=scope + '14')

        fc15 = tf.contrib.layers.fully_connected(fc14, 1024*2 , scope=scope + '15')

        fcc4 = tf.concat([fc13, fc14, fc15], 1)

        fc16 = tf.contrib.layers.fully_connected(fcc4, 1024*2 *2, scope=scope + '16')

        fc17 = tf.contrib.layers.fully_connected(fc16, 1024*2 *2, scope=scope + '17')

        fc18 = tf.contrib.layers.fully_connected(fc17, 1024*2*2 , scope=scope + '18')

        fcc5 = tf.concat([fc16, fc17, fc18], 1)

        fc19 = tf.contrib.layers.fully_connected(fcc5, 1024 *2, scope=scope + '19')

        fc20 = tf.contrib.layers.fully_connected(fc19, 1024*2 , scope=scope + '20')

        fc21 = tf.contrib.layers.fully_connected(fc20, 1024*2 , scope=scope + '21')

        fc22 = tf.contrib.layers.fully_connected(fc21, 1024 *2, scope=scope + '22')

        fcc6 = tf.concat([fc19, fc20, fc21, fc22], 1)

        fc23 = tf.contrib.layers.fully_connected(fcc6, 1024*2 ,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                scope=scope + '23')  # ,normalizer_fn=tf.contrib.layers.batch_norm

        fc24 = tf.contrib.layers.fully_connected(fc23, 1024*2 , scope=scope + '24')

        fc25 = tf.contrib.layers.fully_connected(fc24, 1024 *2, scope=scope + '25')

        fc26 = tf.contrib.layers.fully_connected(fc25, 1024 *2, scope=scope + '26')

        fcc7 = tf.concat([fc24, fc25, fc26], 1)

        fc27 = tf.contrib.layers.fully_connected(fcc7, 1024 *2*2, scope=scope + '27')

        fc28 = tf.contrib.layers.fully_connected(fc27, 1024 *2*2, scope=scope + '28')

        fc29 = tf.contrib.layers.fully_connected(fc28, 1024 *2*2, scope=scope + '29')

        fcc8 = tf.concat([fc27, fc28, fc29], 1)

        fc30 = tf.contrib.layers.fully_connected(fcc8, 1024*2 , scope=scope + '30')

        fc31 = tf.contrib.layers.fully_connected(fc30, 1024*2 , scope=scope + '31')

        fc32 = tf.contrib.layers.fully_connected(fc31, 1024*2 , scope=scope + '32')

        fc33 = tf.contrib.layers.fully_connected(fc32, 1024*2 , scope=scope + '33')

        fcc9 = tf.concat([fc30, fc31, fc32, fc33], 1)

        fc34 = tf.contrib.layers.fully_connected(fcc9, 1,normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.sigmoid, scope=scope + '34')

        return fc34
'''

'''
conv1 = tf.contrib.layers.conv2d(inputx, 4, [8, 8], 1, padding='SAME', activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'conv1')
        conv2 = tf.contrib.layers.conv2d(conv1, 4, [16, 16], [1,1], padding='SAME',activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'conv2')
        #pool1 = (tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"))

        x1 = tf.concat([conv2, conv2], 1)
        y1 = tf.concat([x1, x1], 2)

        conv3 = tf.contrib.layers.conv2d(y1, 8, [8, 8], 1, padding='SAME', activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'conv3')
        conv4 = tf.contrib.layers.conv2d(conv3,8,  [16, 16], [1,1], padding='SAME', activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'conv4')
        #pool2 = (tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"))

        x2 = tf.concat([conv4, conv4], 1)
        y2 = tf.concat([x2, x2], 2)

        conv5 = tf.contrib.layers.conv2d(y2,16,  [8, 8], 1, padding='SAME', activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'conv5')
        conv6 = tf.contrib.layers.conv2d(conv5, 16, [16, 16], [1,1], padding='SAME', activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'conv6')
        #pool3 = (tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"))

        x3 = tf.concat([conv6, conv6], 1)
        y3 = tf.concat([x3, x3], 2)

        conv7 = tf.contrib.layers.conv2d(y3,16,  [8, 8], 1, padding='SAME', activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'conv7')
        conv8 = tf.contrib.layers.conv2d(conv7,16,  [16 ,16], [1,1], padding='SAME', activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'conv8')
        #pool4 = tf.nn.softplus(tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"))




        start=tf.reshape(conv8 , [batchsize, -1])

        fc0 = tf.contrib.layers.fully_connected(start, dim , activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc0')
        fc1 = tf.contrib.layers.fully_connected(fc0, dim, activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc1')


        fc2 = tf.contrib.layers.fully_connected(fc1, dim , activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc2')
        input = fc2
        stat = fc0
        last = fc1
        ht, ct = memroy(stat, input, last,'fc2')
        fc3 = tf.contrib.layers.fully_connected(ct, dim , activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc3')
        input = fc3
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last,'fc3')
        fc4 = tf.contrib.layers.fully_connected(ct, dim , activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc4')
        input = fc4
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last,'fc4')

        fc5 = tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc5')
        input = fc5
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last,'fc5')
        fc6 = tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc6')
        input = fc6
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last,'fc6')
        fc7 = tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc7')
        input = fc7
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last,'fc7')
        fc8 = tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc8')
        input = fc8
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last,'fc8')
        fc9 = tf.contrib.layers.fully_connected(ct, dim, activation_fn=tf.nn.elu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fc9')
        input = fc9
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last,'fc9')

        fcc3 = tf.contrib.layers.fully_connected(ct,dim , activation_fn=tf.nn.elu,
                                                 normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + 'fcc3')
        input = fcc3
        stat = ht
        last = ct
        ht, ct = memroy(stat, input, last,'fcc3')

        fc12 = tf.contrib.layers.fully_connected(ct, 1, activation_fn=tf.nn.elu,
                                                 normalizer_fn=tf.contrib.layers.batch_norm,scope=scope + '12')
                                                 '''
