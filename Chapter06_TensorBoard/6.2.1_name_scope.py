# -*- coding:utf-8 -*-
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# 一层结构的softmax_layer
with tf.name_scope('softmax_layer'):
    weights = tf.Variavle(tf.zeros(784, 10))
    biases = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, weights) + biases
    
    
  