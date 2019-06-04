# -*- coding: utf-8 -*-
import tensorflow as tf
# 创建文件名队列filename_queue
filename_queue = tf.train.string_input_producer(['stat.tfrecord'])
# 创建读取TFRecords文件的reader
reader = tf.TFRecordReader()
# 取出stat.tfrecord文件中的一条序列化的样例serialized_example
_, serialized_example = reader.read(filename_queue)
# 将一条序列化的样例转换为其包含的所有特征张量
features = tf.parse_single_example(
        serialized_example,
        features={
            'id': tf.FixedLenFeature([], tf.int64),
            'age': tf.FixedLenFeature([], tf.int64),
            'income': tf.FixedLenFeature([], tf.float32),
            'outgo': tf.FixedLenFeature([], tf.float32),
        }
)


#================“创建样例队伍”代码补充部分===================
init_op = tf.global_variables_initializer
sess = tf.Session()
sess.run(init_op)
# 启动执行入队操作的后台线程
tf.train.start_queue_runners(sess=sess)
# 读取第一条数据记录
for i in range(2):
    example = sess.run(features)
    print(example)