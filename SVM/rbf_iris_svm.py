# coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess=tf.Session()

iris=datasets.load_iris()
x_vals=np.array([[x[0], x[3]] for x in iris.data])
y_vals1=np.array([1 if y==0 else -1 for y in iris.target])
y_vals2=np.array([1 if y==1 else -1 for y in iris.target])
y_vals3=np.array([1 if y==2 else -1 for y in iris.target])
y_vals =np.array([y_vals1, y_vals2, y_vals3])
class1_x=[x[0] for i,x in enumerate(x_vals) if iris.target[i]==0]
class1_y=[x[1] for i,x in enumerate(x_vals) if iris.target[i]==0]
class2_x=[x[0] for i,x in enumerate(x_vals) if iris.target[i]==1]
class2_y=[x[1] for i,x in enumerate(x_vals) if iris.target[i]==1]
class3_x=[x[0] for i,x in enumerate(x_vals) if iris.target[i]==2]
class3_y=[x[1] for i,x in enumerate(x_vals) if iris.target[i]==2]

batch_size=50
# Initialize placeholders
# 数据集的维度在变化，从单类目标分类到三类目标分类。
# 我们将利用矩阵传播和reshape技术一次性计算所有的三类SVM。
# 注意，由于一次性计算所有分类，
# y_target占位符的维度是[3，None]，模型变量b初始化大小为[3，batch_size]
x_data=tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target=tf.placeholder(shape=[3, None], dtype=tf.float32)
prediction_grid=tf.placeholder(shape=[None, 2], dtype=tf.float32)

# create variables for svm
b=tf.Variable(tf.random_normal(shape=[3, None]))

# Gaussian kernel 
gamma=tf.constant(-10.0)
dist=tf.reduce_sum(tf.square(x_data), 1)
dist=tf.reshape(dist, [-1, 1])
sq_dists=tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
my_kernel=tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Declare function to do reshape/batch multiplication
# 最大的变化是批量矩阵乘法。
# 最终的结果是三维矩阵，并且需要传播矩阵乘法。
# 所以数据矩阵和目标矩阵需要预处理，比如xT·x操作需额外增加一个维度。
# 这里创建一个函数来扩展矩阵维度，然后进行矩阵转置，
# 接着调用TensorFlow的tf.batch_matmul（）函数
def reshape_matmul(mat):
	v1=tf.expand_dims(mat, 1)
	v2=tf.reshape(v1, [3, batch_size, 1])
	return tf.matmul(v2, v1)

first_term = tf.reduce_sum(b)
b_vec_cross= tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target)

second_term = tf.reduce_mean(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# Gaussian (RBF) prediction kernel
# 现在创建预测核函数。
# 要当心reduce_sum（）函数，这里我们并不想聚合三个SVM预测，
# 所以需要通过第二个参数告诉TensorFlow求和哪几个
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1,1])
pred_sq_dist=tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel=tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

# 实现预测核函数后，我们创建预测函数。
# 与二类不同的是，不再对模型输出进行sign（）运算。
# 因为这里实现的是一对多方法，所以预测值是分类器有最大返回值的类别。
# 使用TensorFlow的内建函数argmax（）来实现该功能
