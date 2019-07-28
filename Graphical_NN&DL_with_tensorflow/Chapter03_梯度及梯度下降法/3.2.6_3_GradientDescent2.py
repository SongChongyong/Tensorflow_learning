# coding:utf-8
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 二元函数 f(x_1,x_2)=x_1^2+x_2^2 使用梯度下降法最优化，初始值x^{(1)=(-4,4)}, 学习率设置为0.25
x = tf.Variable(tf.constant([-4,4],tf.float32), dtype=tf.float32)

y= tf.reduce_sum(tf.square(x))

# 梯度下降，学习率设置为0.25
optimizer_1 = tf.train.GradientDescentOptimizer(0.25).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(2):
        sess.run(optimizer_1)
        # 打印每次迭代后x的值
        print(sess.run(x))
        
'''
[-2.  2.]
[-1.  1.]
'''
                
        
        