# coding:utf-8
import tensorflow as tf
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 输入层和输出层
x = tf.placeholder(tf.float32,shape=(1,2),name = "inmput")
y = tf.placeholder(tf.float32,shape=(1,2),name = "outmput")

# 第一层的权重
w1 = tf.constant(
    [[1,4,7],
     [2,6,8]
    ],tf.float32
    )

# 第一层的偏置
b1 = tf.constant(
    [
    [-4,2,1],
    ],tf.float32
    )

# 计算第一层的线性组合
l1 = tf.matmul(x,w1)+b1
# 激活第一层
sigmal1 = 2 * l1


# 第二层偏置
w2 = tf.constant(
    [
    [2,3],
    [1,-2],
    [-1,1]
    ],tf.float32
    )
#第二层偏置
b2 = tf.constant(
    [
    [5,-3]
    ],tf.float32)
# 计算第二层的线性组合
l2 = tf.matmul(sigmal1,w2)+b2
# 激活第二层
sigmal2 = 2*l2

with tf.Session() as sess:
    y = sess.run(sigmal2,feed_dict = {x:np.array([[3,5]],np.float32)})
    print(y)
# [[10. -2.]]


   