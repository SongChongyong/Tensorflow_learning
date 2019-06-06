# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

'''
可视化数据流图可以简化为以下三个步骤：
(1) 创建数据流图
(2) 创建FileWriter实例
(3) 启动TensorBoard程序
'''

#==================(1) 创建数据流图 ==================
from tensorflow.examples.tutorials.mnist import input_data

# 获取MNIST数据集
mnist = input_data.read_data_sets('./mnist', one_hot=True)

# 输入模块
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# softmax网络层
with tf.name_scope('softmax_layer'):
  with tf.name_scope('weights'):
      weights = tf.Variable(tf.zeros([784, 10]))
  with tf.name_scope('biases'):
      biases = tf.Variable(tf.zeros([10]))
  with tf.name_scope('Wx_plus_b'):
      y = tf.matmul(x, weights) + biases

# 交叉熵  
with tf.name_scope('cross_entropy'):
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)

#优化器
with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(0.001).minimize(
      cross_entropy)

#准确率
with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#==================(2) 创建FileWriter实例 ==================
'''
创建FileWriter实例，将(1)中创建的数据流图写入事件中
FileWriter（）中第一个参数logdir表示要创建事件文件的目录，必须设置，
例：writer = tf.summary.FileWriter('./mnist_board')，其中'./mnist_board'就是事件文件的目录
'''
# 创建交互式会话
sess = tf.InteractiveSession()
# 创建FileWriter实例，并传入当前会话加载的数据流图
writer = tf.summary.FileWriter('./mnist_board', sess.graph)
# 初始化全局变量
tf.global_variables_initializer().run()
# 关闭FileWriter的输出流
writer.close()


#==================(3) 启动TensorBoard程序 ==================
# 启动TensorBoard程序，并设置logdir参数为(2)中定义的事件文件目录
# $ tensorboard --logdir=./mnist_board

'''
输出结果为：
TensorBoard 1.9.0 at http://terence-virtual-machine:6006 (Press CTRL+C to quit)
打开浏览器，进入http://terence-virtual-machine:6006就可以看到数据流图，保存结果见'./Pictures_of_Chapter06/6.2.2_MNIST_softmax.png'
'''
