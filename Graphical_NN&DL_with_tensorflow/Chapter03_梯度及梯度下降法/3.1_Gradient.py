# coding:utf-8
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 计算F(x_1,x_2)=(3x_1+4x_2)^2 在x=(2,3)处的梯度值

x = tf.placeholder(tf.float32,(2,1))
w = tf.constant([[3,4]],tf.float32)
y = tf.matmul(w,x)
F = tf.pow(y,2)

grads =tf.gradients(F,x)

with tf.Session() as sess:
    print(sess.run(grads, {x:np.array([[2], [3]])}))
    
'''
[array([[108.],
       [144.]], dtype=float32)]  
'''
             