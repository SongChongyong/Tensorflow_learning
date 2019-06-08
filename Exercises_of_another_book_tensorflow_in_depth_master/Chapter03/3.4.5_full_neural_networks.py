import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 声明两个变量,变量初始值是满足标准差为1,平均值mean为0(默认)
# seed设置种子为1,保证程序运行每次取值固定
w1 = tf.Variable(tf.random_normal((2,3), stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal((3,1), stddev=1, seed=1), name='w2')

# ================================================placeholder 占位
# 定义placeholder作为存放输入数据的地方
# 在shape的一个维度上使用None,可以方便使用不同的batch大小.
# (1)在训练时,需要把数据分成较小的batch; (2)在测试时,可以一次性使用全部数据.
x = tf.placeholder(tf.float32, shape=(None,2), name = 'x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

# 前向传播过程: a = x点乘w1, y = a点乘w2
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# =================================================定义损失函数和反向传播算法
# 使用sigmoid函数将y转换为0~1之间的数值.转换后y代表预测是正样本的概率,1-y代表预测是负样本的概率.
y = tf.sigmoid(y)
# 定于损失函数
cross_entropy = -tf.reduce_mean(
	y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
	+(1-y_)*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# 定义反向传播算法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# ==================================================随机数生成模拟数据集(样本和标签)
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义规则来给出样本的标签:
#	(1)对 x1+x2 < 1, 被认为是正样本, y=1;
#	(2)对 x1+x2 >= 1, 被认为是负样本, y=0;
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]


# 进入会话
with tf.Session() as sess:
	# 变量必须初始化
	# 直接使用 tf.global_variables_initializer().run() 初始化所有变量
	tf.global_variables_initializer().run() 
	# ================================================打印保存训练前初始参数
	# 在训练之前神经网络参数的值
	print("在训练之前神经网络参数的值:")
	print(sess.run(w1))
	print(sess.run(w2))
	
	# 定于训练轮数
	STEPS = 5000
	for i in range (STEPS):
		# 每次选取batch_size个样本进行训练
		start = (i * batch_size) % dataset_size
		end = min(start+batch_size, dataset_size)
		
		# 将选取的训练样本{x,y_}喂入神经网络train_step, 更新参数
		sess.run(train_step,
				feed_dict={x: X[start:end], y_:Y[start:end]})
		
		if i % 1000 ==0:
			# 每隔1000个样本打印一次在所有数据上的交叉熵
			total_cross_entropy = sess.run(cross_entropy,
										feed_dict={x:X, y_:Y})
			print("训练%d次后,交叉熵cross entropy的值是:%g"%(i,total_cross_entropy))
			
	#  ================================================打印保存训练后参数
	print("\n训练之后神经网络的参数为:")
	print(sess.run(w1))
	print(sess.run(w2))

# =============================================训练结果
'''
在训练之前神经网络参数的值:
[[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
[[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
训练0次后,交叉熵cross entropy的值是:0.130842
训练1000次后,交叉熵cross entropy的值是:0.0273095
训练2000次后,交叉熵cross entropy的值是:0.0135175
训练3000次后,交叉熵cross entropy的值是:0.00824951
训练4000次后,交叉熵cross entropy的值是:0.00553266

训练之后神经网络的参数为:
[[-2.435156   2.98222    2.8039565]
 [-3.9361618  1.4693342  3.249323 ]]
[[-2.1775174]
 [ 3.1744573]
 [ 2.312998 ]]

'''
# 随着训练的进行,交叉熵越来越小,说明预测的结果y和实际结果y_差距越来越小,训练有效.
	

#保存计算图
#使用tf.get_default_graph()保存默认的TensorBoard,事件文件保存在'./Chapter03_Board'下
#而后可以在命令行输入: tensorboard --logdir=./Chapter03_Board 启动tensorboard,
#然后在浏览器中查看张量的计算图(见3.4.5_full_neural_networks.png)
writer = tf.summary.FileWriter(logdir='./Chapter03_Board',graph=tf.get_default_graph())
writer.flush()
