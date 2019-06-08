import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 

mnist=input_data.read_data_sets('/home/ajay/scikit_learn_data/tf_mnist', one_hot=True)
train_img=mnist.train.images
train_label=mnist.train.labels
test_img=mnist.test.images
test_lable=mnist.test.labels

x=tf.placeholder("float", [None, 784])
y=tf.placeholder("float", [None, 10])
W=tf.Variable(tf.zeros([784, 10]))
b=tf.Variable(tf.zeros([10]))

# logistic model
actv=tf.nn.softmax(tf.matmul(x,W) + b)
# loss 
loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))
# optimizer
learning_rate=0.01
optm=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# prediction
pred=tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# accuracy
accr=tf.reduce_mean(tf.cast(pred, "float"))

init=tf.global_variables_initializer()
training_echo=50
batch_size=100
display_step=5
num_batch=int(mnist.train.num_examples/batch_size)

sess=tf.Session()
sess.run(init)

for epoch in range(training_echo):
	avg_cost=0.
	for i in range(num_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		feeds_train = {x:batch_xs, y:batch_ys}
		sess.run(optm, feed_dict=feeds_train)
		avg_cost += sess.run(loss, feed_dict=feeds_train) / num_batch

	if epoch % display_step == 0:
		feeds_test = {x:mnist.test.images, y:mnist.test.labels}
		train_acc = sess.run(accr, feed_dict=feeds_train)
		test_acc = sess.run(accr, feed_dict=feeds_test)
		print("Epoch: %03d/%03d Loss: %.9f train_acc: %.3f test_acc: %.3f" 
			% (epoch, training_echo, avg_cost, train_acc, test_acc))

print "Done -----"