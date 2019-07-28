# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

x = tf.Variable(15.0, dtype=tf.float32)

y= tf.pow(x-1, 2.0)

# 梯度下降法求  f(x) = (x-1)^2 的最小值点，学习率设置为0.25
optimizer_2 = tf.train.GradientDescentOptimizer(0.05).minimize(y)

# 画曲线
value = np.arange(-15,17,0.01)
y_value = np.power(value-1,2.0)
plt.plot(value, y_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(100):
        sess.run(optimizer_2)
        if (i%10==0):
            v = sess.run(x)
            plt.plot(v, math.pow(v-1, 2.0), "go")
            # 打印每次迭代后x的值
            print("第%d次的x的迭代值：%f" %(i+1,v))
        

plt.show()               

'''
第1次的x的迭代值：13.600000
第11次的x的迭代值：5.393349
第21次的x的迭代值：2.531866
第31次的x的迭代值：1.534129
第41次的x的迭代值：1.186239
第51次的x的迭代值：1.064938
第61次的x的迭代值：1.022642
第71次的x的迭代值：1.007895
第81次的x的迭代值：1.002753
第91次的x的迭代值：1.000960
'''        
        