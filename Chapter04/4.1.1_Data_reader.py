# coding:utf-8
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

'''
4.1.1 使用流水线并行读取数据
工作流程：  
（1） 创建文件名列表  
（2）创建文件名队伍  
（3）创建Reader和Decoder  
（4）创建样例队伍  
'''

#===================== (1) 创建文件名列表 =====================
# 创建文件名列表有两种方法：
#    （1）当文件名个数少时，使用python列表，如['file0.csv','file1.csv']
#     (2) 使用tf.train.match_filenames_once方法


#===================== (2) 创建文件名队伍  =====================
# 使用tf.train.string_input_producer方法
# tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True, 
#                               seed=None, capacity=32,shared_name=None,name=None,cancel_op=None)
# 其中string_tensor表示“存储文件名列表的字符串张量”，num_epochs表示“最大训练周期”
# shuffle表示“是否打乱文件名顺序”，seed表示“随机化种子”


#===================== (3) 创建Reader和Decoder  =====================
# Reader功能是读取数据记录，Decoder的功能是将数据记录转换为张量格式
# (1)对CSV文件，Reader: tf.TextLineReader,  Decoder: tf.decode_csv
# (2)对TFRecores文件，RTeader:tf.TFRecordReader , Decorder:tf.parse_single_example

# filename_queue = tf.train.string_input_producer(['stat0.csv','stat1.csv'])
# 
# reader = tf.TextLineReader()
# 
# value = reader.read(filename_queue)
# 
# record_defaults = [[0],[0],[0.0],[0.0]]
# 
# id,age,income,outgo = tf.decode_csv(value,
#                                     record_defaults=record_defaults)
# features = tf.stack([id,age,income,outgo])
# 
# with tf.Session as sess:
#     print(features.eval())

# 另见4.1_writer.py & 4.1_writer.py

#===================== (4) 创建样例队伍  =====================
# 使用tf.train.start_queue_runners方法启动执行入队操作的所有线程
# 直接使用tf.train.start_queue_runners方法时，任何线程发生错误，程序都会崩溃，
# 为此，使用tf.train.Cordinator方法创建管理多线程的协调器