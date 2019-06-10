import tensorflow as tf
from numpy.random import RandomState
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# =========================== 1. 定义变量及滑动平均类==========================
# 定义一个变量用于计算滑动平均,变量的初始值都为0
v1 = tf.Variable(0, dtype=tf.float32)
# 定义step变量模拟神经网络中迭代的轮数,可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类,初始化时给定衰减率class为0.99, 控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新滑动平均的操作
maintain_averages_op = ema.apply([v1]) 

# ==============================2. 查看不同迭代中变量取值的变化==============================
with tf.Session() as sess:
    
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print (sess.run([v1, ema.average(v1)]))             # [0.0, 0.0]
    
    # =====================================更新变量v1 
    # 更新变量v1的取值为5
	# v1的滑动平均会被更新为0.1*0+0.9*5=4.5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均: (1)衰减率为min{0.99, (1+step)/(10+step)}=0.1
    #                (2)v1的滑动平均会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print (sess.run([v1, ema.average(v1)]))                 # [5.0, 4.5]
    
    # ===================================== 更新step和v1的取值
	# 更新后step的值为10000, v1的值为10
    sess.run(tf.assign(step, 10000))  
    sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均: (1)衰减率为min{0.99, (1+step)/(10+step)}=0.99
    #                (2)v1的滑动平均会被更新为0.99*4.5+0.01*10=4.555
    sess.run(maintain_averages_op)
    print (sess.run([v1, ema.average(v1)]))                 # [10.0, 4.555]
    
    # 更新一次v1的滑动平均值, v1=4.555
    # 更新v1的滑动平均: (1)衰减率为min{0.99, (1+step)/(10+step)}=0.99
    #                (2)v1的滑动平均会被更新为0.99*4.555+0.01*10=4.60945
    sess.run(maintain_averages_op)
    print (sess.run([v1, ema.average(v1)]))                 # [10.0, 4.60945]
'''
[0.0, 0.0]
[5.0, 4.5]
[10.0, 4.555]
[10.0, 4.60945]
'''      
