import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 声明两个变量,变量初始值是满足标准差为1,平均值mean为0(默认)
# seed设置种子为1,保证程序运行每次取值固定
w1 = tf.Variable(tf.random_normal((2,3), stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal((3,1), stddev=1, seed=1), name='w2')

# ===========================================placeholder 占位
# 定义placeholder作为存放输入数据的地方
# 这样在会话中,可以生成大量常量数据来提供给输入x
x = tf.placeholder(tf.float32, shape=(3,2), name = 'input')

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
	# ===========================================feed_dict字典传入输入数据x
	# feed_dict字典传入输入数据, 打印运算结果y,
	print(sess.run(y, feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
	

#保存计算图
#使用tf.get_default_graph()保存默认的TensorBoard,事件文件保存在'./Chapter03_Board'下
#而后可以在命令行输入: tensorboard --logdir=./Chapter03_Board 启动tensorboard,
#然后在浏览器中查看张量的计算图(见3.4.4_Placeholder.png)
writer = tf.summary.FileWriter(logdir='./Chapter03_Board',graph=tf.get_default_graph())
writer.flush()
