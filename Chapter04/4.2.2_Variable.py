# -*- coding: utf-8 -*-
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 创建、初始化、更新模型参数例子


#创建变量 
W = tf.Variable(0.0, name='W')
double = tf.multiply(2.0, W)

with tf.Session() as sess:
    #初始化全局变量
    sess.run(tf.global_variables_initializer())
    #循环执行4次加法赋值
    for i in range(4):
        #加法赋值，更新新的W
        sess.run(tf.assign_add(W, 1.0))
        print('W=%s, double=%s'%(sess.run(W),sess.run(double)))
        

'''
输出结果是：
W=1.0, double=2.0
W=2.0, double=4.0
W=3.0, double=6.0
W=4.0, double=8.0     
'''