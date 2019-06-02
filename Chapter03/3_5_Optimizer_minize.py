# -*- coding=utf-8 -*-
import tensorflow as tf

# 优化器是Tensorflow实现优化算法的载体，它为用户实现了自动计算模型参数梯度值的功能
# Tensorflow提供了常用的各种优化器: Gradient Descent, Adam, Momentum, ...
# 模型训练的过程需要最小化损失函数，为此，TensorFlowde 所有优化器均实现了
# 用于最小化损失函数的minimize方法


# ============以梯度下降优化器为例，使用minimize方法训练模型典型步骤==============
# 模型
X = tf.placeholder(...)
W = tf.Variable(...)
b = tf.Variable(...)

Y = tf.matmul(X, W) + b
Y_ = tf.placeholder(...)

# 使用交叉熵作为损失函数
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))

# 优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# global_step 表示全局训练步数，随着模型迭代优化自增，即为0, 0+1, 0+1+1, ...
global_step = tf.Variable(0, name = 'global_step', trainable=False)
# 最小化损失值
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer()) 
    for step in range(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y_: train_Y})
        # 训练日志：每隔log_step步打印一次日志
        if step % log_step == 0:
            c = sess.run(loss, feed_dict={X: train_X, Y_:train_Y})
            print("Step:%d, loss==%.4f, W==%.4f, b==%.4f" % 
                    (step, c, sess.run(W), sess.run(b)))
    # 计算训练完毕的模型在训练集上的损失值，作为指标输出
    final_loss = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
    # 计算训练完毕的模型参数W和b
    weight, bias = sess.run([W, b])
    print("Step:%d, loss==%.4f, W==%.4f, b==%.4f" % 
            (max_train_steps, final_loss, sess.run(W), sess.run(b)))
    print("Linear Regression Model: Y==%.4f*X+%.4f" % (weight, bias))

