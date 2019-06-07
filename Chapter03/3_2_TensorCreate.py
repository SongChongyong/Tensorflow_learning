import tensorflow as tf

#==========================================（1）
# 张量的创建包括： （1）常量的定义操作；（2）代数计算操作
a = tf.constant(1.0)        #创建常量
b = tf.constant(2.0)        #add操作
c = tf.add(a,b)

print([a,b,c])

#输出结果为：[<tf.Tensor 'Const:0' shape=() dtype=float32>, 
#             <tf.Tensor 'Const_1:0' shape=() dtype=float32>, 
#             <tf.Tensor 'Add:0' shape=() dtype=float32>]
# 由输出结果知道，Tensorflow计算的是张量的结构，而不是具体的数字
# 一个张量中主要保存了三个属性：name, shape, dtype         


#==========================================（2）
# （2）张量的求解
# 数据流图中操作输出值由张量承载，如果需要求解特定张量的值，需要创建会话
# 然后执行张量的eval方法或会话的run方法
with tf.Session() as sess:
    print(c.eval())
    print(sess.run([a,b,c]))

# 输出结果为：
# 3.0
# [1.0, 2.0, 3.0]


#保存计算图
#使用tf.get_default_graph()保存默认的TensorBoard,事件文件保存在'./test_board'下
#而后可以在命令行输入: tensorboard --logdir=./test_board 启动tensorboard,
#然后在浏览器中查看张量的计算图(见3_2_TensorCreate_Board.png)
writer = tf.summary.FileWriter(logdir='./test_board',graph=tf.get_default_graph())
writer.flush()

