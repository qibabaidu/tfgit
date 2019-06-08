import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from sklearn import datasets

sess=tf.Session()
iris=datasets.load_iris()
x_vals=np.array([x[3] for x in iris.data])
y_vals=np.array([y[0] for y in iris.data])
batch_size=50
learning_rate=0.1
iterations=250
x_data=tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target=tf.placeholder(shape=[None, 1], dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))
model_output=tf.add(tf.matmul(x_data, A), b)

# demming
demming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

init=tf.global_variables_initializer()
sess.run(init)
my_opt_L1=tf.train.GradientDescentOptimizer(learning_rate)
train_setp_L1=my_opt_L1.minimize(loss)
loss_vec_L1=[]

for i in range(iterations):
	rand_index=np.random.choice(len(x_vals), size=batch_size)
	rand_x=np.transpose([x_vals[rand_index]])
	rand_y=np.transpose([y_vals[rand_index]])
	sess.run(train_setp_L1, feed_dict={x_data: rand_x, y_target:rand_y})
	temp_loss_L1=sess.run(loss, feed_dict={x_data: rand_x, y_target:rand_y})
	loss_vec_L1.append(temp_loss_L1)
	if (i+1)% 50 == 0:
		print "Step #" + str(i+1) + " A="+ str(sess.run(A))+ " b=" + str(sess.run(b))
		print "Loss = " + str(temp_loss_L1)

[slope] = sess.run(A)
[y_intercept] = sess.run(b)
best_fit = []
for i in x_vals:
	best_fit.append(slope*i + y_intercept)
plt.plot(x_vals, y_vals, 'o', label="Data Points")
plt.plot(x_vals, best_fit, 'r-', label="Best fit line")
plt.show()