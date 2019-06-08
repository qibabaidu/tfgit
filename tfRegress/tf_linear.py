import tensorflow as tf 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data_set=pd.read_csv("train.csv")
data_set=data_set[data_set["LotArea"]<12000]

data_X=data_set["LotArea"].values.reshape(-1,1)
data_Y=data_set["SalePrice"].values.reshape(-1,1)

n_samples=data_X.shape[0]
learning_rate=2
training_epochs=1000
display_step=50

dict_X=tf.placeholder(tf.float32)
dict_Y=tf.placeholder(tf.float32)

# y=w*x+b
w=tf.Variable(np.random.randn(), name="weight", dtype=tf.float32)
b=tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)

# define module
y_pred=tf.add(tf.multiply(w, dict_X) ,b)
# define cost
cost=tf.reduce_sum(tf.pow(y_pred-dict_Y, 2))/(2*n_samples)
# optimization problem => optimize
optimizer= tf.train.AdamOptimizer(learning_rate).minimize(cost)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        for (x,y) in zip(data_X, data_Y):
            sess.run(optimizer, feed_dict={dict_X:x, dict_Y:y})
          
        if(epoch+1)%display_step==0:
            c=sess.run(cost, feed_dict={dict_X:data_X, dict_Y:data_Y})
            print "Epoch: ","%04d" % (epoch+1), "cost=","{:.3f}".format(c), "W=",sess.run(w), "b=",sess.run(b)

    print "Optimization Finished!"
    training_cost=sess.run(cost, feed_dict={dict_X:data_X, dict_Y:data_Y})
    print "Traing cost=", training_cost, "W=",sess.run(w), "b=",sess.run(b),"\n"

    # plot
    plt.plot(data_X, data_Y, "ro", label="Original data")
    plt.plot(data_X, sess.run(w)*data_X+sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()

sess.close()


