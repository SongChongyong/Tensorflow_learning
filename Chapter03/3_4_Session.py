# coding: utf-8
import tensorflow as tf
import numpy as np

# 会话提供求解张量和执行操作的运行环境
# 会话的典型使用流程是：
#        (1) 创建会话 sess = tf.Session()
#        (2) 运行会话 sess.run
#        (3) 关闭会话 sess.close()

#==========================================(1) 创建会话 sess = tf.Session()=======================
# (1) 创建会话 sess = tf.Session()
# tf.Session(target, graph, config)
# target：会话连接的执行引擎，默认值指向 in-process engine
# gragh： 会话加载的数据流图
# config: 会话启动时的配置项


#==========================================(2) 运行会话 sess.run=======================
# (2) 运行会话 sess.run
# sess.run(fetchs, feed_dict, options, run_metadata)
# 参数fetchs: 带求解的张量或操作
# 参数 feed_dict: 数据填充字典，形如<数据节点，填充数据>
# 求解张量，可以用两种方法： a. Session.run()  b.张量的eval方法

x = tf.placeholder(tf.float32)
w = tf.Variable(1.0)
b = tf.Variable(1.0)

y = w*x+b 

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    # 用Session.run()求解张量
    print(sess.run(y, feed_dict={x:5.0}))
    
    # 用张量的eval方法求解张量
    fetch = y.eval(feed_dict={x:3.0})
    print(fetch)


#==========================================(3) 关闭会话 =======================
# (3) 关闭会话 
# 有两种关闭会话的方法：
#     a. 使用close方法: sess.close()
#     b. 使用with语句隐式关闭会话----with语句会隐式调用Session.__exit__方法，自动关闭会话
   
