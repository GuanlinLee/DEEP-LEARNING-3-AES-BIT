import tensorflow as tf
import  numpy as np
import new_data_input as data_input
import new_branch_2 as model
from time import  time

batchsize=4
length=16
CHECKFILE = './checkpoint/model.ckpt'
lr=3e-3
decay=0.8
max_grad_norm=10
N=8
classes=2

def train1(max_step):
    global lr
    inputx = tf.placeholder(tf.float32, shape=[batchsize, 8], name="inputx")
    inputy = tf.placeholder(tf.int32, shape=[batchsize ], name='inputy')

    one_hot = tf.one_hot(inputy, classes)

    outputy = model.model(inputx,batchsize=batchsize,is_train=1)
    realout=outputy
    #realout=tf.reshape(tf.convert_to_tensor(outputy[:,0]),[batchsize,1])

    correct_prediction = tf.equal(tf.argmax(one_hot, 1), tf.argmax(outputy, 1))
    cast=tf.cast(correct_prediction, tf.int32)
    acc = tf.reduce_sum(tf.cast(correct_prediction, tf.int32),0)/batchsize
    # loss = tf.reduce_sum(tf.where(tf.greater(tf.reduce_sum(outputy), tf.reduce_sum(answer)),
    #                           (tf.reduce_sum(outputy)-tf.reduce_sum(answer)), (tf.reduce_sum(answer)-tf.reduce_sum(outputy))))
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=outputy),0)/batchsize+1e-6
    # loss = tf.reduce_sum(tf.where(tf.greater(tf.reduce_sum(outputy), tf.reduce_sum(answer)),
    #                           (tf.reduce_sum(outputy)-tf.reduce_sum(answer)), (tf.reduce_sum(answer)-tf.reduce_sum(outputy))))
    #loss = tf.losses.mean_squared_error(tf.multiply(inputy, 1), tf.multiply(realout, 1))/batchsize
   # tvars = tf.trainable_variables()
   # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),max_grad_norm)

    cesummary = tf.summary.scalar(name='loss', tensor=loss)
    accsummary = tf.summary.scalar(name='acc', tensor=acc)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step=tf.train.AdamOptimizer(lr).minimize(loss)#apply_gradients(zip(grads, tvars))

    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./checkpoint/')
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('start training from new state')
        summary_writer = tf.summary.FileWriter('./tensorboard/', sess.graph)
        for step in range(start_step, start_step + max_step):
            global_step=step
            start_time = time()
            x,xy=data_input.inputdata(batchsize,step)
            sess.run(train_step, feed_dict={inputx: x,inputy:xy})
            summary_all = sess.run(merged, feed_dict={inputx: x,inputy:xy})
            summary_writer.add_summary(summary_all, step)
            real=sess.run(realout, feed_dict={inputx: x,inputy:xy})
            realindex=np.argmax(real,1)
            equalnp=np.equal(xy,realindex)
            if step % 10 == 0:

                train_loss = sess.run(loss, feed_dict={inputx: x,inputy:xy})
                duration = time() - start_time
                print('step:',step,'   loss:',train_loss,'     spend_time:',duration)
                print(sess.run(tf.argmax(one_hot, 1), feed_dict={inputx: x, inputy: xy}))
                print(sess.run(tf.argmax(outputy, 1), feed_dict={inputx: x, inputy: xy}))
#                print(sess.run(correct_prediction,feed_dict={inputx:x,inputy:xy}))
                print(sess.run(acc, feed_dict={inputx: x,inputy:xy}))
                print(sess.run(outputy, feed_dict={inputx: x,inputy: xy}))
#                print(realindex)
#                print(equalnp)
            if step % 60 == 1:
                lr = lr * decay
                saver.save(sess, CHECKFILE, global_step=step)
                print('writing checkpoint at step %s' % step)
        summary_writer.close()


def test1():
    inputx = tf.placeholder(tf.float32, shape=[batchsize, 8], name="inputx")
    inputy=tf.placeholder(tf.int32,shape=[batchsize],name='inputy')

    outputy = model.model(inputx,batchsize=batchsize,is_train=0)
    realout = outputy
    one_hot = tf.one_hot(inputy, classes)

    # loss = tf.reduce_sum(tf.where(tf.greater(tf.reduce_sum(outputy), tf.reduce_sum(answer)),
    #                           (tf.reduce_sum(outputy)-tf.reduce_sum(answer)), (tf.reduce_sum(answer)-tf.reduce_sum(outputy))))
    #loss = tf.losses.mean_squared_error(tf.multiply(inputy, 0.1), tf.multiply(realout, 0.1))/batchsize
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=outputy))
    correct_prediction = tf.equal(tf.argmax(one_hot, 1), tf.argmax(outputy, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accnum = 0.0
    sumof = 0.0
    acclist = []

    # loss = tf.reduce_sum(tf.where(tf.greater(tf.reduce_sum(outputy), tf.reduce_sum(answer)),
    #                           (tf.reduce_sum(outputy)-tf.reduce_sum(answer)), (tf.reduce_sum(answer)-tf.reduce_sum(outputy))))
    #loss=tf.multiply(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot,logits=outputy)),tf.reduce_sum(tf.losses.mean_squared_error(tf.multiply(one_hot,20),tf.multiply(outputy,20))))



    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./checkpoint/')
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            raise Exception('no checkpoint find')


        for t in range(N//batchsize):

            start_time = time()
            x,xy = data_input.testdata(batchsize,t)

            if t % 1 == 0:
                train_loss = sess.run(loss, feed_dict={inputx: x,inputy:xy})
                duration = time() - start_time
                real = sess.run(realout, feed_dict={inputx: x, inputy: xy})
                pred = sess.run(outputy, feed_dict={inputx: x, inputy: xy})
                for p in range(batchsize):
                    if pred[p][0] != 0.5:
                        sumof += 1.0
                        if pred[p][0] > pred[p][1] and xy[p] == 0:
                            accnum += 1
                        elif pred[p][0] < pred[p][1] and xy[p] == 1:
                            accnum += 1.0

                print('step:', t, '   loss:', train_loss, '     spend_time:', duration)
                print((sess.run(inputy, feed_dict={inputx: x, inputy: xy})))
                print((sess.run(realout, feed_dict={inputx: x, inputy: xy})))

                acclist.append(sess.run(acc, feed_dict={inputx: x, inputy: xy}))

    accnp = np.array(acclist)
    print(accnp)
    print(np.mean(accnp))
    print(sumof, '   ', accnum, '    ', accnum / sumof)
