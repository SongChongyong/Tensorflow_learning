# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist', one_hot=True)

with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('softmax_layer'):
  weights = tf.Variable(tf.zeros([784, 10]))
  biases = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, weights) + biases
  
with tf.name_scope('cross_entropy'):
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
    #添加获取交叉熵的汇总操作
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(0.001).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #添加获取准确率的汇总操作
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
sess = tf.InteractiveSession()

# 创建FileWriter实例，并传入当前会话加载的数据流图
# 数据流图写入事件文件,分别保持在./mnist_scalar/train和./mnist_scalar/test文件夹下
train_writer = tf.summary.FileWriter('./mnist_scalar' + '/train', sess.graph)
test_writer = tf.summary.FileWriter('./mnist_scalar' + '/test')
tf.global_variables_initializer().run()

def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs, ys = mnist.train.next_batch(100, fake_data=False)
  else:
    xs, ys = mnist.test.images, mnist.test.labels
  return {x: xs, y_: ys}

for i in range(1000):
  if i % 10 == 0:  
    # Record summaries and test-set accuracy 获取汇总数据和准确率 
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    if i % 100 == 99:  # Record execution stats
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step],
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
      print('Adding run metadata for', i)
    else:  # Record a summary 向事件文件中写入汇总数据，并传入全局步数
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()

'''
输出结果为：
Accuracy at step 0: 0.098
Accuracy at step 10: 0.5948
Accuracy at step 20: 0.7267
Accuracy at step 30: 0.7655
Accuracy at step 40: 0.7913
Accuracy at step 50: 0.8125
Accuracy at step 60: 0.821
Accuracy at step 70: 0.8185
Accuracy at step 80: 0.8313
Accuracy at step 90: 0.8355
Adding run metadata for 99
Accuracy at step 100: 0.8489
...
Accuracy at step 980: 0.9133
Accuracy at step 990: 0.9138
Adding run metadata for 999
'''

# $ tensorboard --logdir=./mnist_scalar     启动TensorBoard，而后进入浏览器查看生成的折线图
