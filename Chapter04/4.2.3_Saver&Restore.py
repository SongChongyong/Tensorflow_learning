# -*- coding: utf-8 -*-
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#4.2.3 使用tf.train.Saver保存和恢复模型参数
# 创建saver步骤： 
# (1) 创建Saver
# (2) 保存为ckpt文件
# (3) 训练时恢复保存的变量



#创建变量 
W = tf.Variable(0.0, name='W')
double = tf.multiply(2.0, W)

#
saver = tf.train.Saver({'weights':W})


with tf.Session() as sess:
    #初始化全局变量
    sess.run(tf.global_variables_initializer())
    
    #循环执行4次加法赋值
    for i in range(4):
        #加法赋值，更新新的W
        sess.run(tf.assign_add(W, 1.0))
        print('W=%s, double=%s'%(sess.run(W),sess.run(double)))
        #存储变量W
        saver.save(sess, './Saver/4.2.3.ckpt')
        
'''
输出结果是：
W=1.0, double=2.0
W=2.0, double=4.0
W=3.0, double=6.0
W=4.0, double=8.0     
'''
        
#=================== 恢复模型参数 ========================        
with tf.Session() as sess:
    #初始化全局变量
    saver.restore(sess, './Saver/4.2.3.ckpt')
    print('Restored:W=%s'%sess.run(W))
    
    #循环执行4次加法赋值
    for i in range(4):
        #加法赋值，更新新的W
        sess.run(tf.assign_add(W, 1.0))
        print('W=%s, double=%s'%(sess.run(W),sess.run(double)))

'''
输出结果是：
Restored:W=4.0
W=5.0, double=10.0
W=6.0, double=12.0
W=7.0, double=14.0
W=8.0, double=16.0
'''