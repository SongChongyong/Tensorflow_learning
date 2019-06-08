import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 声明两个变量,变量初始值是满足标准差为1,平均值mean为0(默认)
# seed设置种子为1,保证程序运行每次取值固定
w1 = tf.Variable(tf.random_normal((2,3), stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal((3,1), stddev=1, seed=1), name='w2')

# 将输入特征定义为一个常量
x = tf.constant([[0.7, 0.9]], name = 'x')

# 前向传播 a = x点乘w1, y = a点乘w2
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# 进入会话
with tf.Session() as sess:
	# 变量必须初始化
	# 可以逐个变量初始化,使用initializer方法
	# 或者直接使用 tf.global_variables_initializer().run() 初始化所有变量
	sess.run(w1.initializer)
	sess.run(w2.initializer)
	# 打印运算结果y
	print(sess.run(y))
	

#保存计算图
#使用tf.get_default_graph()保存默认的TensorBoard,事件文件保存在'./Chapter03_Board'下
#而后可以在命令行输入: tensorboard --logdir=./Chapter03_Board 启动tensorboard,
#然后在浏览器中查看张量的计算图(见3.4.3_Variable.png)
writer = tf.summary.FileWriter(logdir='./Chapter03_Board',graph=tf.get_default_graph())
writer.flush()
