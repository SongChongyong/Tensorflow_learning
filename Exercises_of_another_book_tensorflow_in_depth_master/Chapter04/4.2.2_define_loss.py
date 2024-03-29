import tensorflow as tf
from numpy.random import RandomState
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义一个单层的前向传播的神经网络
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                             (y - y_) * loss_more,
                             (y_ - y) * loss_less))

train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集 
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            print("After %d training step(s), w1 is: " % (i))
            print (sess.run(w1))
            print("\n")
    print ("Final w1 is: \n" + str(sess.run(w1)))
        