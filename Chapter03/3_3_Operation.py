# coding: utf-8
import tensorflow as tf
import numpy as np

# Tnesorflow的算法模型由数据流图表示，数据流图由节点和有向边组成，每个节点均对应一个具体的操作。节点按照功能分为：
# 计算节点 Operation
# 存储节点 Variable
# 数据节点 Placeholder

#==========================================(1) 计算节点 Operation=======================
# (1) 计算节点 Operation
# 每个计算节点对输入张量进行特定的数学运算或流程控制，然后将结果输出到后置的节点。
# 典型的计算操作：
#    (1) 基础算术： add, multiply, mod, sqrt
#    (2) 初始化操作： zaero_initializer, random_normal_initializer
#    (3) 神经网络运算: convolution, pool, softmax, dropout
#    (4) 随机运算： random_normal, random_shuffle

#==========================================(2) 存储节点 Variable=======================
# (2) 存储节点 Variable----变量是有状态节点，其内部的变量操作长期保存变量的对应的值
# 作用是： 在多次执行相同数据流图时存储特定的参数
# 一个变量通常由以下四个子节点构成：
#    (1) 变量初始值 intial_value
#    (2) 更新变量值的操作 Assign
#    (3) 读取变量值的操作 Read
#    (4) 变量操作  如：(a)





#==========================================(3) 数据节点 Placeholder=======================
# (3) 数据节点 Placeholder
# 作用：定义待输入数据的属性 (在创建模型的阶段，用户不需要向数据流图输入任何数据)
# 数据节点由占位符操作(placeholder operation) 实现，对应的函数操作是：
# tf.placeholder(dtype, shape, name)


with tf.name_scope('PlaceholderExample'):
    x = tf.placeholder(tf.float32,shape=(2,2),name="X")
    y = tf.matmul(x,x,name="matmul")
    
with tf.Session() as sess:
    rand_array = np.random.rand(2,2)
    print("rand_array is :")
    print(rand_array)
    print("Y is :")
    print(sess.run(y,feed_dict={x:rand_array}))
 

