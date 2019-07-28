# coding:utf-8
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 梯度下降法求  f(x) = (x-1)^2 的最小值点，学习率设置为0.25
x = tf.Variable(4.0, dtype=tf.float32)

y= tf.pow(x-1, 2.0)

# 梯度下降，学习率设置为0.25
optimizer_1 = tf.train.GradientDescentOptimizer(0.25).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(3):
        sess.run(optimizer_1)
        # 打印每次迭代后x的值
        print(sess.run(x))
        
'''
2.5
1.75
1.375
'''
                
        
        